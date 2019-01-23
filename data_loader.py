# original code is from https://github.com/aaron-xichen/pytorch-playground
import torch
import os
import numpy.random as nr
import numpy as np
import bisect

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_raw_data(target_data_loader, num_classes, num_val):
    flag = 0
    total_data, total_label = 0, 0
    validation_index = []
    test_index = []
    
    label_count = np.empty(num_classes)
    label_count.fill(num_val)
    
    for data, target in target_data_loader:
        data, target = data.cuda(), target.cuda()
        if flag == 0:
            total_label = target
            total_data = data
        else:
            total_label = torch.cat((total_label, target), 0)
            total_data = torch.cat((total_data, data), 0)
        flag = 1
    
    for index in range(0, total_data.size(0)):
        label = total_label[index]
        if label_count[label] > 0:
            validation_index.append(index)
            label_count[label] -= 1
        else:
            test_index.append(index)
            
    return total_data, total_label, validation_index, test_index

def get_validation(total_data, total_label, validation_index, test_index):
    output = []
    output.append(total_data.index_select(0, torch.LongTensor(validation_index).cuda()))
    output.append(total_label.index_select(0, torch.LongTensor(validation_index).cuda()))
    output.append(total_data.index_select(0, torch.LongTensor(test_index).cuda()))
    output.append(total_label.index_select(0, torch.LongTensor(test_index).cuda()))
    
    return output

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train_shuffle=True, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
                #target_transform=target_transform,
            ),
            batch_size=batch_size, shuffle=train_shuffle, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
                #target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train_shuffle=True, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=train_shuffle, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train_shuffle=True, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=train_shuffle, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, input_TF, dataroot, train_shuffle = True):
    if data_type == 'cifar10':
        train_loader, test_loader \
        = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, train_shuffle=train_shuffle, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader \
        = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, train_shuffle=train_shuffle, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader \
        = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, train_shuffle=train_shuffle, num_workers=1)

    return train_loader, test_loader
