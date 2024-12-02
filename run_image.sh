python train_classification_image.py \
--epoch 300 \
--batch_size 1024 \
--learning_rate 0.005 \
--gpu 0 \
--folder image \
--c_prune_rate 2.5 \
--num_feat 2048 \
--dataset FMNIST \
--num_category 10 \
--model model_cls_rand_mnist \
--use_normals \
--sparsity 0.5 \
--scale 1. \
--quantize full \
--r0 0.15 \
--r1 0.2 \
--hard_mode batch