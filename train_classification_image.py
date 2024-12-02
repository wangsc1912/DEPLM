import os
import sys
import torch
from torch import nn
import numpy as np
import torchvision

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import torchvision.transforms as transforms
# import tonic
# import tonic.transforms as transforms
from utility import utils
from utility.image_to_point import toPoint, toPointMnist

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.model_utils import sparse_weight_gen, model_weight_gen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='model_cls_rand_mnist', choices=['model_cls_rand', 'model_cls_rand_mnist'], help='model name [default: pointnet_cls]')
    parser.add_argument('--dataset', default='FMNIST', choices=['FMNIST'], help='dataset name')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 11, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--folder', default='random', help='experiment root')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--r0', type=float, default=0.2, help='radius 0')
    parser.add_argument('--r1', type=float, default=0.4, help='radius 0')

    parser.add_argument('--sa_iter', type=str, default='1,1,1', help='grouping model recurrent times')
    parser.add_argument('--num_feat', type=int, default=2048)
    parser.add_argument('--noise', type=float, default=0., help='noise level')  # noise per vector-matrix multiplication
    parser.add_argument('--c_prune_rate', type=float, default=1, help='channel pruning ratio')
    parser.add_argument('--quantize', type=str, default='full', choices=['full'], help='quantize weight or not')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity of mixture normal')
    parser.add_argument('--scale', type=float, default=0.001, help='scale of mixture normal')
    parser.add_argument('--distance', type=str, default='l2', choices=['l1', 'l2'], help='type of distance for grouping')
    parser.add_argument('--num_fc', type=int, default=1, help='number of FC layer')
    parser.add_argument('--normal_feature', type=int, default=3, help='number of normal feature')

    parser.add_argument('--trainable', action='store_true', default=False, help='trainable')
    parser.add_argument('--hard_mode', type=str, default='batch', choices=[None, 'vmm', 'batch'])
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if args.dataset == 'cifar10':
        exp_dir = Path('./log/cifar10')
    elif args.dataset == 'MNIST':
        exp_dir = Path('./log/mnist')
    elif args.dataset == 'FMNIST':
        exp_dir = Path('./log/fmnist')
    else:
        exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    if args.dataset == 'FMNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            toPointMnist(28, 28, 1, args.normal_feature)
            ])
        train_dataset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST('./data', train=False,
                transform=transform)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/model_utils.py', str(exp_dir))
    shutil.copy('./train_classification_image.py', str(exp_dir))

    if args.model in ['model_cls_rand', 'model_cls_ssg_mnist']:
        sa_iter = list(map(int, args.sa_iter.split(',')))
        classifier = model.get_model(num_class, normal_feature=args.normal_feature, c_prune_rate=args.c_prune_rate,
                                     iter=sa_iter, noise=args.noise, quantize=args.quantize,
                                     num_feat=args.num_feat, distancing=args.distance, num_fc=args.num_fc,
                                     r0=args.r0, r1=args.r1, hard_mode=args.hard_mode)
    else:
        classifier = model.get_model(num_class, normal_channel=args.use_normals, c_prune_rate=args.c_prune_rate)
    
    # sortware weight sparsity
    classifier, cond_dict = utils.replace_model_weight(classifier, args.sparsity)

    # freeze the conv layers
    if not args.trainable:
        for name, params in classifier.named_parameters():
            if 'conv' in name or 'bn' in name:
                params.requires_grad = False

    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    # number of parameter
    num_param_hard, num_train_param = 0, 0
    for name, params in classifier.named_parameters():
        if ('bn' not in name) and ('fc' not in name):
            num_param_hard += params.numel()
        elif 'fc' in name:
                num_train_param += params.numel()
    log_string(f'Number of hardware weights: {num_param_hard}, Software readout: {num_train_param}')

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
 
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cond_dict': cond_dict
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)