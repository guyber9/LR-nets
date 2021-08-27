from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import os

from models import *
from utils import find_sigm_weights, train, test, print_summary, copy_net2net

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def main_train():
    # TODO
    # lr 0.1 or 0.01
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
    parser.add_argument('--mnist', action='store_true', default=False, help='mnist flag')
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='M',
                        help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--full-prec', action='store_true', default=False, help='For Training Full Precision Model')
    parser.add_argument('--load-pre-trained', action='store_true', default=False,
                        help='For Loading Params from Trained Full Precision Model')
    parser.add_argument('--debug-mode', action='store_true', default=False, help='For Debug Mode')
    parser.add_argument('--parallel-gpu', type=int, default=1, metavar='N', help='parallel-gpu (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='num_workers (default: 4)')
    parser.add_argument('--num', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--wd', type=int, default=4, metavar='N', help='wd is 10**((-1)*wd)')
    parser.add_argument('--pd', type=int, default=11, metavar='N', help='pd is 10**((-1)*pd)')
    parser.add_argument('--bn-wd', type=int, default=4, metavar='N', help='pd is 10**((-1)*bn_wd)')
    parser.add_argument('--binary-mode', action='store_true', default=False, help='binary mode bit')
    parser.add_argument('--nohup', action='store_true', default=False, help='nohup mode')
    parser.add_argument('--dont-save', action='store_true', default=False, help='dont_save mode')

    parser.add_argument('--sgd', action='store_true', default=False, help='run with sgd')

    parser.add_argument('--save-file', action='store', default='no_need_to_save', help='name of saved model')

    parser.add_argument('--debug', action='store_true', default=False, help='run with adam')

    parser.add_argument('--options', type=int, default=1, metavar='N', help='num_of_options for rand')
    parser.add_argument('--tickets', type=int, default=1, metavar='N', help='num of tickets')
    parser.add_argument('--sampled-test', action='store_true', default=False, help='sampled validation in training')

    parser.add_argument('--add-bn-bias-decay', action='store_true', default=False,
                        help='dd weight decay to bn(weight/bias) and bias')

    parser.add_argument('--ver2', action='store_true', default=False, help='discretization for layer output')
    parser.add_argument('--cudnn', action='store_true', default=False, help='using cudnn benchmark=True')
    parser.add_argument('--collect_stats', action='store_true', default=False, help='collect_stats for test')

    parser.add_argument('--suffix', action='store', default='', help='suffix for saved model name')
    parser.add_argument('--annealing-sched', action='store_true', default=False, help='using CosineAnnealingLR')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    best_epoch = 0
    best_sampled_acc = 0  # best test accuracy
    best_sampled_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.save_file != 'no_need_to_save':
        writer_suffix = '_' + str(args.save_file)
    else:
        writer_suffix = ''
    writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_ver2/' if args.ver2 else '/') + str(
        writer_suffix)
    #     writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_new_run')
    writer = SummaryWriter(writer_name)

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
            root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    else:
        print("############################")
        print("## no data set was chosen ##")
        print("############################")
        exit(1)

    # Model
    print('==> Building model..')
    if args.cifar10:
            print("Training LR-Net for CIFAR10 | ver2")
            net = LRNet_CIFAR10_ver2(writer)
            net.load_state_dict(torch.load('saved_models/cifar10_lrnet_ver2_debug_2.pt.pt.pt'))

    if device == 'cuda':
        if args.cudnn:
            print('==> Using cudnn.benchmark = True')
            cudnn.benchmark = True
        else:
            print('==> Using cudnn.benchmark = False && torch.backends.cudnn.deterministic = True')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    weight_decay = 10 ** ((-1) * args.wd)
    criterion = nn.CrossEntropyLoss()

    if args.cifar10:
        print('==> Building CIFAR10 | ver2 optimizer')
        optimizer = optim.Adam([
            {'params': net.fc1.parameters()},
            {'params': net.fc2.parameters()}
        ], lr=args.lr, weight_decay=weight_decay)

    net = net.to(device)

    if args.save_file != 'no_need_to_save':
        file_name = "tmp_logs/" + str(args.save_file) + ".log"
        f = open(file_name, "w")
        print(args, file=f)
    else:
        print(args)
        f = None

    for epoch in range(start_epoch, start_epoch + args.epochs):
        net.test_mode_switch(1, 1)
        # net.train_mode_switch()
        # net.tensorboard_train = True
        train_acc = train(net, criterion, epoch, device, trainloader, optimizer, args, f, writer)
        writer.add_scalar("acc/train", train_acc, epoch)
        # net.tensorboard_train = False

        best_acc, best_epoch, test_acc = test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch,
                                              test_mode=False, f=f, eval_mode=True,
                                              dont_save=True)  # note: model is saved only in test method below
        writer.add_scalar("cont acc/test", test_acc, epoch)

        net.test_mode_switch(1, 1)

        # net.tensorboard_test = True
        best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader, args,
                                                                 best_sampled_acc, best_sampled_epoch, test_mode=False,
                                                                 f=f, eval_mode=True, dont_save=False, writer=writer)
        writer.add_scalar("sampled_acc/test", sampled_acc, epoch)
        # net.tensorboard_test = False

        print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)

        writer.add_scalar("sampled_acc/test", sampled_acc, epoch)

        scheduler.step()

    # if args.ver2:
    #     net = net.to('cpu')
    #     torch.save(net.state_dict(), "saved_models/debug_cpu.pt")
    # #             torch.save(net.state_dict(), "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + str(args.suffix) + ".pt")

    writer.flush()
    writer.close()

    if args.save_file != 'no_need_to_save':
        f.close()


if __name__ == '__main__':
    main_train()

