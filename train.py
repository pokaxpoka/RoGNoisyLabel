"""
Created on Jan 23 2019
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import models
import os

from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch code: icml submission 2243')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='./parameters/', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='40,80', help='decreasing strategy')
parser.add_argument('--net_type', default='densenet', help="Type of Classification Nets")
parser.add_argument('--optimizer_flag', default='sgd', help="Type of optimizer")
parser.add_argument('--numclass', type=int, default=10, help='the # of classes')
parser.add_argument('--noise_fraction', type=int, default=0, help='noisy fraction')
parser.add_argument('--label_root', default='./labels/', help='folder to labels')
parser.add_argument('--noise_type', default='uniform', help='type_of_noise')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ', args.dataset)

transform_train = transforms.Compose([
    transforms.RandomCrop(args.imageSize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])

if args.dataset == 'cifar100':
    args.numclass = 100
    args.decreasing_lr = '80,120,160'
    args.epochs = 200

train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, transform_train, args.dataroot)
_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, transform_test, args.dataroot)

if args.noise_fraction > 0:
    print('load noisy labels')
    args.label_root = args.label_root  + args.dataset +'/' + args.noise_type  + '/' + str(args.noise_fraction)
    if os.path.isdir(args.label_root) == False:
        print('Err: generate noisy labels first')
    else:
        args.label_root = args.label_root  + '/train_labels.npy'
        
    new_label = torch.load(args.label_root)
    train_loader.dataset.train_labels = new_label
    
print('Model: ', args.net_type)
if args.net_type == 'densenet':
    model = models.DenseNet3(100, int(args.numclass))
elif args.net_type == 'resnet34':
    model = models.ResNet34(num_c=args.numclass)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)                
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), total,
                100. * batch_idx / total, loss.data[0]))

def test(epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data), dim=1)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= args.droprate
    test(epoch)
    
args.outf = args.outf + '/' + args.net_type + '/' + args.dataset + '/' + args.noise_type + '/' + str(args.noise_fraction) + '/'

if os.path.isdir(args.outf) == False:
    os.makedirs(args.outf)
torch.save(model.state_dict(), '%s/model.pth' % (args.outf))