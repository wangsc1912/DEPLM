import torch
import tonic
import sys
sys.path.append('.')
tr_dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
te_dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
event, target = tr_dataset[10]

print(tr_dataset)

