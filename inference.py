"""
Created on Jan 23 2019
"""
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader
import numpy as np
import models
import math
import os
import generative_utils as utils
import scipy
import torch.nn as nn
import numpy.random as nr

from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch code: icml submission 2243')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--outf', default='./parameters/', help='folder to output images and model checkpoints')
parser.add_argument('--net_type', default='densenet', help="Type of Classification Nets")
parser.add_argument('--noise_fraction', type=int, default=0, help='noisy fraction')
parser.add_argument('--label_root', default='./labels/', help='folder to labels')
parser.add_argument('--noise_type', default='uniform', help='type_of_noise')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

batch_size = 200
args.label_root = args.label_root + args.dataset + '/' + args.noise_type + '/' + str(args.noise_fraction) + '/train_labels.npy'
file_root \
= args.outf + '/' + args.net_type + '/' + args.dataset + '/' + args.noise_type + '/' + str(args.noise_fraction) + '/'

num_output = 4
if args.net_type == 'densenet':
    num_output = 3

layer_list = list(range(num_output))

torch.cuda.manual_seed(0)
torch.cuda.set_device(args.gpu)

print('load dataset: '+ args.dataset)
num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100
in_transform = transforms.Compose([transforms.ToTensor(), \
                                   transforms.Normalize((125.3/255, 123.0/255, 113.9/255), \
                                                        (63.0/255, 62.1/255.0, 66.7/255.0)),])
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, batch_size, in_transform, args.dataroot, False)

if args.noise_fraction > 0:
    print('load noisy labels')
    new_label = torch.load(args.label_root)
    train_loader.dataset.train_labels = new_label
        
num_val = 50
if args.dataset == 'cifar100':
    num_val = 5
total_train_data, total_train_label, _, _ = data_loader.get_raw_data(train_loader, num_classes, 0)

total_test_data, total_test_label, val_index, test_index \
= data_loader.get_raw_data(test_loader, num_classes, num_val)

total_val_data, total_val_label, total_test_data, total_test_label \
= data_loader.get_validation(total_test_data, total_test_label, val_index, test_index)

print('load networks: '+ args.net_type)
if args.net_type == 'resnet34':
    model = models.ResNet34(num_c=num_classes)
elif args.net_type == 'densenet':
    model = models.DenseNet3(100, int(num_classes))
model.load_state_dict(torch.load(file_root + 'model.pth', map_location = "cuda:" + str(args.gpu)))
model.cuda()

print('extact features')
utils.extract_features(model, total_train_data, total_train_label, file_root, "train_val")
utils.extract_features(model, total_val_data, total_val_label, file_root, "test_val")
utils.extract_features(model, total_test_data, total_test_label, file_root, "test_test")

test_data_list, test_label_list = [], []
test_val_data_list, test_val_label_list = [], []
train_data_list, train_label_list = [], []

for layer in layer_list:    
    file_name_data = '%s/test_test_feature_%s.npy' % (file_root, str(layer))
    file_name_label = '%s/test_test_label.npy' % (file_root)

    test_data = torch.from_numpy(np.load(file_name_data)).float()
    test_label = torch.from_numpy(np.load(file_name_label)).long()
    test_data_list.append(test_data)

    file_name_data = '%s/test_val_feature_%s.npy' % (file_root, str(layer))
    file_name_label = '%s/test_val_label.npy' % (file_root)

    test_data_val = torch.from_numpy(np.load(file_name_data)).float()
    test_label_val = torch.from_numpy(np.load(file_name_label)).long()
    test_val_data_list.append(test_data_val)
    file_name_data = '%s/train_val_feature_%s.npy' % (file_root, str(layer))
    file_name_label = '%s/train_val_label.npy' % (file_root)
    
    if args.noise_type == 'uniform':
        for index in range(int(test_label_val.size(0)*args.noise_fraction/100)):
            prev_label = test_label_val[index]
            while(True):
                new_label = nr.randint(0, num_classes)
                if prev_label != new_label:
                    test_label_val[index] = new_label
                    break;
                    
    elif args.noise_type == 'flip':
        for index in range(int(test_label_val.size(0)*args.noise_fraction/100)):
            prev_label = test_label_val[index]
            new_label = (prev_label + 1) % num_classes
            test_label_val[index] = new_label
            
    train_data = torch.from_numpy(np.load(file_name_data)).float()
    train_label = torch.from_numpy(np.load(file_name_label)).long()

    train_data_list.append(train_data)
    train_label_list.append(train_label)

test_label_list.append(test_label)
test_val_label_list.append(test_label_val)

print('Random Sample Mean')
sample_mean_list, sample_precision_list = [], []
for index in range(len(layer_list)):
    sample_mean, sample_precision, _ = \
    utils.random_sample_mean(train_data_list[index].cuda(), train_label_list[index].cuda(), num_classes)
    sample_mean_list.append(sample_mean)
    sample_precision_list.append(sample_precision)

print('Single MCD and merge the parameters')
new_sample_mean_list = []
new_sample_precision_list = []
for index in range(len(layer_list)):
    new_sample_mean = torch.Tensor(num_classes, train_data_list[index].size(1)).fill_(0).cuda()
    new_covariance = 0
    for i in range(num_classes):
        index_list = train_label_list[index].eq(i)
        temp_feature = train_data_list[index][index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        temp_mean, temp_cov, _ \
        = utils.MCD_single(temp_feature.cuda(), sample_mean_list[index][i], sample_precision_list[index])
        new_sample_mean[i].copy_(temp_mean)
        if i  == 0:
            new_covariance = temp_feature.size(0)*temp_cov
        else:
            new_covariance += temp_feature.size(0)*temp_cov
            
    new_covariance = new_covariance / train_data_list[index].size(0)
    new_precision = scipy.linalg.pinvh(new_covariance)
    new_precision = torch.from_numpy(new_precision).float().cuda()
    new_sample_mean_list.append(new_sample_mean)
    new_sample_precision_list.append(new_precision)

G_soft_list = []
target_mean = new_sample_mean_list 
target_precision = new_sample_precision_list
for i in range(len(new_sample_mean_list)):
    dim_feature = new_sample_mean_list[i].size(1)
    sample_w = torch.mm(target_mean[i], target_precision[i])
    sample_b = -0.5*torch.mm(torch.mm(target_mean[i], target_precision[i]), \
                             target_mean[i].t()).diag() + torch.Tensor(num_classes).fill_(np.log(1./num_classes)).cuda()
    G_soft_layer = nn.Linear(int(dim_feature), num_classes).cuda()
    G_soft_layer.weight.data.copy_(sample_w)
    G_soft_layer.bias.data.copy_(sample_b)
    G_soft_list.append(G_soft_layer)

print('Construct validation set')
sel_index = -1
selected_list = utils.make_validation(test_val_data_list[sel_index], test_val_label_list[-1], \
                                target_mean[sel_index], target_precision[sel_index], num_classes)
new_val_data_list = []
for i in range(len(new_sample_mean_list)):
    new_val_data = torch.index_select(test_val_data_list[i], 0, selected_list.cpu())
    new_val_label = torch.index_select(test_val_label_list[-1], 0, selected_list.cpu())
    new_val_data_list.append(new_val_data)
        
soft_weight = utils.train_weights(G_soft_list, new_val_data_list, new_val_label)
soft_acc = utils.test_softmax(model, total_test_data, total_test_label)

RoG_acc = utils.test_ensemble(G_soft_list, soft_weight, test_data_list, test_label_list[-1])

print('softmax accuracy: ' + str(soft_acc))
print('RoG accuracy: '+ str(RoG_acc))
