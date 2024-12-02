import torch
from torch.utils.data import Dataset
import h5py
import os
import sys
sys.path.append('.')
import numpy as np
import random
import pickle


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


class DvsDataset(Dataset):
    def __init__(self, DATADIR, train, num_points=1024, use_raw=True, sample='random_sample'):
        super(DvsDataset, self).__init__()
        self.num_points = num_points
        self.use_raw = use_raw
        self.sample = sample

        if self.use_raw:
            self.dataset_dir = os.path.join(DATADIR, "train") if train else os.path.join(DATADIR, "test")
            files = os.listdir(self.dataset_dir)
            print("processing dataset:{} ".format(self.dataset_dir))
        else:
            files = getDataFiles(os.path.join(DATADIR, 'train_files.txt')) if train else getDataFiles(os.path.join(DATADIR, 'test_files.txt'))
            print("processing dataset:{} ".format(DATADIR))

        self.data, self.label = [], []
        if self.use_raw:
            for f in files:
                with open(os.path.join(self.dataset_dir, f), 'rb') as f:
                    dataset = pickle.load(f)
                self.data += dataset['data']
                self.label += dataset['label'].tolist()
        else:
            for f in files:
                d, l= loadDataFile(os.path.join(DATADIR, f))
                self.data.append(d)
                self.label.append(l)
            self.data = np.concatenate(self.data, axis=0).squeeze()
            self.label = np.concatenate(self.label, axis=0).squeeze()
        print(train, len(self.data))


    def random_sample(self, events):
        nr_events = events.shape[0]
        idx = np.arange(nr_events) 
        np.random.shuffle(idx)
        idx_full = np.zeros(self.num_points, dtype=int)
        if nr_events <= self.num_points:
            idx_full[0: nr_events] = idx[0: nr_events]
        else:
            idx_full = idx[0: self.num_points]
        # idx = idx[0: self.num_points]   
        events = events[idx_full, ...]
        return events

    def continue_sample(self, events):
        total_events = events.shape[0]
        if (total_events <= self.num_points):
            start_i = 0
            end_i = total_events
            valid_length = total_events
        else:
            start_i = np.random.randint(0, total_events - self.num_points)
            end_i = start_i + self.num_points
            valid_length = self.num_points

        idx_full = np.zeros(self.num_points, dtype=int)
        idx_full[0 : valid_length] = np.arange(start_i, end_i)
        events = events[idx_full, ...]
        return events   

    def uniform_sample(self, events):
        total_events = events.shape[0]
        if (total_events <= self.num_points):
            start_i = 0
            end_i = total_events
            valid_length = total_events
            idx_full = np.zeros(self.num_points, dtype=int)
            idx_full[0 : valid_length] = np.arange(start_i, end_i)
        else:
            scale = math.floor(total_events / self.num_points)
            start_i = np.random.randint(1, scale+1)
            idx_full = np.arange(self.num_points) * scale
            idx_full = np.clip(idx_full, a_min=0, a_max=total_events)

        events = events[idx_full, ...]
        return events   

    def __getitem__(self, index):
        if self.use_raw:
            label = int(self.label[index])
            events = self.data[index]
            if self.sample == 'random_sample':
                events = self.random_sample(events)
            elif self.sample == 'continue_sample':
                events = self.continue_sample(events)
            elif self.sample == 'uniform_sample':
                events = self.uniform_sample(events)
            else:
                raise ValueError

            events_normed = np.zeros_like(events, dtype=np.float32)
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            events_normed[:, 1] = x / 127
            events_normed[:, 2] = y / 127
            t = t - np.min(t)
            t = t / np.max(t)
            events_normed[:, 0] = t 

            return events_normed, label
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    DATADIR = 'data/DVS_C10_TS1_1024'
    tr = DvsDataset(DATADIR, train=True)
    length = len(tr)
    print(length)
