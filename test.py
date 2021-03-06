import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--nohup', action='store_true', default=True, help='nohup mode')
    parser.add_argument('--dont-save', action='store_true', default=True, help='dont_save mode')

    parser.add_argument('--adam', action='store_true', default=False, help='run with adam')

    parser.add_argument('--save', action='store', default='tmp_models/cifar10', help='name of saved model')

    parser.add_argument('--options', type=int, default=10, metavar='N', help='num_of_options for rand')
    parser.add_argument('--tickets', type=int, default=1, metavar='N', help='num of tickets')
    parser.add_argument('--ver2', action='store_true', default=False, help='discretization for layer output')
    parser.add_argument('--cudnn', action='store_true', default=True, help='using cudnn benchmark=True')

    parser.add_argument('--train-mode', action='store_true', default=False, help='train_mode also for test (collect stats')
    parser.add_argument('--collect-stats', action='store_true', default=False, help='collect_stats for test')

    parser.add_argument('--suffix', action='store', default='', help='suffix for saved model name')

    parser.add_argument('--no-shuffle', action='store_true', default=False, help='shuffle the data')
    parser.add_argument('--writer', action='store_true', default=False, help='collect stats')
    parser.add_argument('--dont-load-model', action='store_true', default=False, help='collect stats')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': not args.no_shuffle}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if device == 'cuda':
        if args.cudnn:
            print('==> Using cudnn.benchmark = True')
            cudnn.benchmark = True
        else:
            print('==> Using cudnn.benchmark = False && torch.backends.cudnn.deterministic = True')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if args.writer:
        writer_a = 'mnist' if args.mnist else 'cifar10'
        writer_b = '_ver2' if args.ver2 else ''
        writer_name = "runs/test_" + str(writer_a) + str(writer_b) + '/' + str(writer_a) + str(writer_b) + str(args.suffix)
        writer = SummaryWriter(writer_name)
    else:
        writer = None
    
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
#             net = LRNet_CIFAR10_ver2(writer)
            net = LRNet_CIFAR10_ver2_not_sample()
        else:
            print ("Testing LR-Net for CIFAR10")
            net = LRNet_CIFAR10(writer)
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

#     dataset_name = 'mnist' if args.mnist else 'cifar10'
#     net_type = '_fp' if args.full_prec else '_lrnet'
#     isBinary = '_binary' if args.binary_mode else ''
#     isVer2 = '_ver2' if args.ver2 else ''
#     is_sampled = "_sampled" if args.collect_stats else ''
#     load_model_name = "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + str(is_sampled) + str(args.suffix) + ".pt"
#     print('==> Loading model: ' + str(load_model_name))
#     if not args.dont_load_model:
#         net.load_state_dict(torch.load(load_model_name))
   
#     load_model_name = 'saved_models/cifar10_not_sample_not_gumbel_30_10_bn_last_layer_sz_cpu.pt'

#     load_model_name = 'saved_models/cifar10_lrnet_ver2no_gumbel_tau_1_0_23_11.pt'
#     net.load_state_dict(torch.load(load_model_name))    
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

    its_eval_mode = not args.train_mode

    print ("###################################")
    print ("Original Trained Model (no ternary)")
    print ("###################################")
    print ("test Data Set")
    net.tensorboard_train = True if args.writer else False
#     test(net, criterion, 0, device, testloader, args, 0, None, test_mode, None, eval_mode=True, dont_save=True)
    net.tensorboard_train = False
    print ("train Data Set")
    test(net, criterion, 0, device, trainloader, args, 0, None, test_mode, None, eval_mode=True, dont_save=True)
    exit(1)

    if not args.full_prec:
        print ("###################################")
        print ("Ternary Model")
        print ("###################################")
        print ("test Data Set")
        for idx in range(0, args.options):
            # .update_use_batch_stats(True)
            print("iteration: " + str(idx))
            net.test_mode_switch(args.options, args.tickets)
            net.tensorboard_test = True if args.writer else False
            print("tensorboard_test: " + str(net.tensorboard_test))
            acc, _, _ = test(net, criterion, 0, device, testloader, args, 0, None, test_mode, None, eval_mode=its_eval_mode, dont_save=True)
            net.tensorboard_test = False                
#             if (acc > best_acc):
#                 best_acc = acc
#                 dataset_name = 'mnist' if args.mnist else 'cifar10'
#                 isBinary = '_binary' if args.binary_mode else '_ternary'
#                 isVer2 = '_ver2' if args.ver2 else ''
#                 torch.save(net.state_dict(),
#                            "trained_models/" + str(dataset_name) + "_lrnet" + str(isBinary) + str(isVer2) + ".pt")
        print ("\n\n==> The best acc is :" + str(best_acc) + "\n\n\n")

        print ("train Data Set")
        # test(net, trainloader)
        test(net, criterion, 0, device, trainloader, args, 0, None, test_mode, None, eval_mode=its_eval_mode, dont_save=True)

        print ("\n\n==> The best acc is :" + str(best_acc) + "\n\n\n")

        if args.writer:
            writer.flush()
            writer.close()
    
if __name__ == '__main__':
    main_test()























