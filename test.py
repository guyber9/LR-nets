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
    parser.add_argument('--dont-save', action='store_true', default=False, help='dont_save mode')

    parser.add_argument('--adam', action='store_true', default=False, help='run with adam')

    parser.add_argument('--save', action='store', default='tmp_models/cifar10', help='name of saved model')

    parser.add_argument('--options', type=int, default=10, metavar='N', help='num_of_options for rand')
    parser.add_argument('--tickets', type=int, default=1, metavar='N', help='num of tickets')
    parser.add_argument('--ver2', action='store_true', default=False, help='discretization for layer output')

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
        elif args.ver2:
            print("Testing LR-Net for CIFAR10 | ver2")
            net = LRNet_CIFAR10_ver2()
        else:
            print ("Testing LR-Net for CIFAR10")
            net = LRNet_CIFAR10()
    elif args.mnist:
        if args.full_prec:
            print ("Testing FP-Net for MNIST")
            net = FPNet().to(device)
        elif args.ver2:
            print("Testing LR-Net for MNIST | ver2")
            net = LRNet_ver2().to(device)
        else:
            print ("Testing LR-Net for MNIST")
            net = LRNet().to(device)

    criterion = nn.CrossEntropyLoss()
    test_mode = True

    # dataset_name = 'mnist' if args.mnist else 'cifar10'
    # isBinary = '_binary' if args.binary_mode else '_ternary'
    # net.load_state_dict(torch.load("trained_models/" + str(dataset_name) + "_lrnet" + str(isBinary) + ".pt"))
    # net.eval()
    # net = net.to(device)
    # best_acc, _ = test(net, criterion, 0, device, testloader, args, 0, None, test_mode)
    # print("\n\n==> The best acc is :" + str(best_acc) + "\n\n\n")
    best_acc = 0

    dataset_name = 'mnist' if args.mnist else 'cifar10'
    net_type = '_fp' if args.full_prec else '_lrnet'
    isBinary = '_binary' if args.binary_mode else ''
    isVer2 = '_ver2' if args.ver2 else ''
    load_model_name = "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + ".pt"
    print('==> Loading model: ' + str(load_model_name))
    net.load_state_dict(torch.load(load_model_name))
    net.eval()
    net = net.to(device)

    # ######################################################################
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    #
    # test(net, criterion, 0, device, trainloader, args, 0, None, test_mode)
    #
    # print("sampled")
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    #
    # net.conv1.test_mode_switch(args.options, args.tickets)
    # net.conv2.test_mode_switch(args.options, args.tickets)
    #
    # test(net, criterion, 0, device, trainloader, args, 0, None, test_mode)
    #
    # exit(1)
    # ######################################################################

    print ("###################################")
    print ("Original Trained Model (no ternary)")
    print ("###################################")
    print ("test Data Set")
    test(net, criterion, 0, device, testloader, args, 0, None, test_mode)
    print ("train Data Set")
    test(net, criterion, 0, device, trainloader, args, 0, None, test_mode)

    if not args.full_prec:
        if args.ver2:
            net.test_mode_switch(args.options, args.tickets)
        elif args.cifar10:
            net.conv1.test_mode_switch(args.options, args.tickets)
            net.conv2.test_mode_switch(args.options, args.tickets)
            net.conv3.test_mode_switch(args.options, args.tickets)
            net.conv4.test_mode_switch(args.options, args.tickets)
            net.conv5.test_mode_switch(args.options, args.tickets)
            net.conv6.test_mode_switch(args.options, args.tickets)
        elif args.mnist:
            net.conv1.test_mode_switch(args.options, args.tickets)
            net.conv2.test_mode_switch(args.options, args.tickets)

        print ("###################################")
        print ("Ternary Model")
        print ("###################################")
        print ("test Data Set")
        for idx in range(0, args.options):
            print("iteration: " + str(idx))
            acc, _, _ = test(net, criterion, 0, device, testloader, args, 0, None, test_mode)
            if args.ver2:
                net.inc_cntr()
            else:
                net.conv1.cntr = net.conv1.cntr + 1
                net.conv2.cntr = net.conv2.cntr + 1
                if args.cifar10:
                    net.conv3.cntr = net.conv3.cntr + 1
                    net.conv4.cntr = net.conv4.cntr + 1
                    net.conv5.cntr = net.conv5.cntr + 1
                    net.conv6.cntr = net.conv6.cntr + 1
            if (acc > best_acc):
                best_acc = acc
                dataset_name = 'mnist' if args.mnist else 'cifar10'
                isBinary = '_binary' if args.binary_mode else '_ternary'
                isVer2 = '_ver2' if args.ver2 else ''
                torch.save(net.state_dict(),
                           "trained_models/" + str(dataset_name) + "_lrnet" + str(isBinary) + str(isVer2) + ".pt")
        print ("\n\n==> The best acc is :" + str(best_acc) + "\n\n\n")

        if args.ver2:
            net.rst_cntr()
        else:
            net.conv1.cntr = 0
            net.conv2.cntr = 0
            if args.cifar10:
                net.conv3.cntr = 0
                net.conv4.cntr = 0
                net.conv5.cntr = 0
                net.conv6.cntr = 0
        print ("train Data Set")
        # test(net, trainloader)
        test(net, criterion, 0, device, trainloader, args, 0, None, test_mode)

        print ("\n\n==> The best acc is :" + str(best_acc) + "\n\n\n")

if __name__ == '__main__':
    main_test()























