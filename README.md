# ICML_submission_2243
Codes for icml submission 2243

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries Pytorch package to be installed:

* [Pytorch](http://pytorch.org/): Only GPU version (0.3.1) is available.
* [scipy](https://github.com/scipy/scipy)
* [scikit-learn](http://scikit-learn.org/stable/)

## Training networks with noisy labels

### 1. Generate noisy labels:
```
# dataset: CIFAR-10, noise type: uniform, noise fraction: 60%
python generate_labels.py --dataset cifar10 --noise_type uniform --noise_fraction 60
```

### 2. Train networks
```
# model: DenseNet, dataset: CIFAR-10, noise type: uniform, noise fraction: 60%, gpu 0
python train.py --net_type densenet --dataset cifar10 --noise_type uniform --noise_fraction 60 --gpu 0
```

## Performance evaluation
```
# model: DenseNet, dataset: CIFAR-10, noise type: uniform, noise fraction: 60%
python inference.py --net_type densenet --dataset cifar10 --noise_type uniform --fraction 60 --gpu 0
```
