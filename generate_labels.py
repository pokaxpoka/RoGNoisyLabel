"""
Created on Jan 23 2019
"""
from __future__ import print_function
import torch
import torch.nn as nn
import data_loader
import numpy as np
import os
import numpy.random as nr
import argparse

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch code: icml submission 2243')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--outf', default='./labels/', help='folder to save label')
parser.add_argument('--noise_fraction', type=int, default=20, help='noise fraction')
args = parser.parse_args()
print(args)

def main():
    # make directory to save labels
    args.outf = args.outf + args.dataset + '/uniform/' + str(args.noise_fraction) + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)

    print('load dataset: '+ args.dataset)
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    in_transform = transforms.Compose([transforms.ToTensor(), \
                                       transforms.Normalize((125.3/255, 123.0/255, 113.9/255),\
                                                            (63.0/255, 62.1/255.0, 66.7/255.0)),])
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, 200, in_transform, args.dataroot)

    # generate index_list to change the labels
    train_label = torch.LongTensor(train_loader.dataset.train_labels)
    total_index = torch.LongTensor(list(range(train_label.size(0))))
    total_selected_index = 0
    for index in range(num_classes):
        index_list = train_label.eq(index)
        num_samples_per_class = sum(index_list)
        num_selected_samples = int(num_samples_per_class*args.noise_fraction/100)
        random_index = torch.randperm(num_samples_per_class)
        selected_index = total_index[index_list][random_index][:num_selected_samples]
        if index == 0:
            total_selected_index = selected_index
        else:
            total_selected_index = torch.cat((total_selected_index, selected_index), 0)

    # change labels
    total_new_label = train_loader.dataset.train_labels

    for index in total_selected_index:
        prev_label = total_new_label[index]
        while(True):
            new_label = nr.randint(0, num_classes)
            if prev_label != new_label:
                total_new_label[index] = new_label
                break;
                
    torch.save(total_new_label, '%s/train_labels.npy' % (args.outf))

if __name__ == '__main__':
    main()