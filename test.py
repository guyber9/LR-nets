'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

import os
import argparse

from models import *
from utils import progress_bar
from utils import find_sigm_weights
from utils import test

def main_test():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
    parser.add_argument('--mnist', action='store_true', default=False, help='mnist flag')
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')

    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='M',help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--full-prec', action='store_true', default=False, help='For Training Full Precision Model')
    parser.add_argument('--load-pre-trained', action='store_true', default=False,help='For Loading Params from Trained Full Precision Model')
    parser.add_argument('--debug-mode', action='store_true', default=False, help='For Debug Mode')
    parser.add_argument('--parallel-gpu', type=int, default=1, metavar='N',help='parallel-gpu (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='num_workers (default: 4)')
    parser.add_argument('--num', type=int, default=4, metavar='N',  help='how many batches to wait before logging training status')
    parser.add_argument('--wd', type=int, default=4, metavar='N', help='wd is 10**((-1)*wd)')
    parser.add_argument('--pd', type=int, default=11, metavar='N', help='pd is 10**((-1)*pd)')
    parser.add_argument('--binary-mode', action='store_true', default=False, help='binary mode bit')
    parser.add_argument('--nohup', action='store_true', default=False, help='nohup mode')

    parser.add_argument('--adam', action='store_true', default=False, help='run with adam')

    parser.add_argument('--save', action='store', default='tmp_models/cifar10', help='name of saved model')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    if args.cifar10:
        print('==> Preparing CIFAR10 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.mnist:
        print('==> Preparing MNIST data..')
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        trainset = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        testset = datasets.MNIST('../data', train=False,
                           transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    else:
        print("############################")
        print("## no data set was chosen ##")
        print("############################")
        exit(1)

    # Model
    print('==> Building model..')
    if args.cifar10:
        if args.full_prec:
            print ("Testing FP-Net for CIFAR10")
            net = FPNet_CIFAR10()
        else:
            print ("Testing LR-Net for CIFAR10")
            net = LRNet_CIFAR10()
    elif args.mnist:
        if args.full_prec:
            print ("Testing FP-Net for MNIST")
            net = FPNet().to(device)
        else:
            print ("Testing LR-Net for MNIST")
            net = LRNet().to(device)

    dataset_name = 'mnist' if args.mnist else 'cifar10'
    net_type = '_fp' if args.full_prec else '_lrnet'
    isBinary = '_binary' if args.binary_mode else ''
    net.load_state_dict(torch.load("saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + ".pt"))
    net.eval()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    print ("###################################")
    print ("Original Trained Model (no ternary)")
    print ("###################################")
    print ("test Data Set")
    test(net, criterion, 0, device, testloader, args, 0, None, True)
    print ("train Data Set")
    test(net, criterion, 0, device, trainloader, args, 0, None, True)

    if not args.full_prec:
        if args.cifar10:
            net.conv1.test_mode_switch()
            net.conv2.test_mode_switch()
            net.conv3.test_mode_switch()
            net.conv4.test_mode_switch()
            net.conv5.test_mode_switch()
            net.conv6.test_mode_switch()
        elif args.mnist:
            net.conv1.test_mode_switch()
            net.conv2.test_mode_switch()

    num_of_options = 30
    print ("###################################")
    print ("Ternary Model")
    print ("###################################")
    print ("test Data Set")
    for idx in range(0, num_of_options):
        print("iteration: " + str(idx))
        test(net, criterion, 0, device, testloader, args, 0, None, True)
        net.conv1.cntr = net.conv1.cntr + 1
        net.conv2.cntr = net.conv2.cntr + 1
    print ("train Data Set")
    # test(net, trainloader)
    test(net, criterion, 0, device, trainloader, args, 0, None, True)

if __name__ == '__main__':
    main_test()























