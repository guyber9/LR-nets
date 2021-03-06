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
import resnet
import resnet1

from models import *
from utils import find_sigm_weights, train, test, print_summary, copy_net2net

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def main_train():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
    parser.add_argument('--mnist', action='store_true', default=False, help='mnist flag')
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')
    parser.add_argument('--cifar100', action='store_true', default=False, help='cifar100 flag')
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

    parser.add_argument('--add-bn-bias-decay', action='store_true', default=False, help='dd weight decay to bn(weight/bias) and bias')

    parser.add_argument('--ver2', action='store_true', default=False, help='discretization for layer output')
    parser.add_argument('--cudnn', action='store_true', default=True, help='using cudnn benchmark=True')
    parser.add_argument('--collect_stats', action='store_true', default=False, help='collect_stats for test')

    parser.add_argument('--suffix', action='store', default='', help='suffix for saved model name')
    parser.add_argument('--annealing-sched', action='store_true', default=False, help='using CosineAnnealingLR')
    parser.add_argument('--freeze', action='store_true', default=False, help='freeze layers')

    parser.add_argument('--lnum', type=int, default=6, metavar='N', help='num of layers')
    parser.add_argument('--step', type=int, default=35, metavar='N', help='step size in freezeing')
    parser.add_argument('--start', type=int, default=50, metavar='N', help='starting point in freezeing')
    parser.add_argument('--trials', type=int, default=15, metavar='N', help='num of trials in freezeing')

    parser.add_argument('--only-1-fc', action='store_true', default=False, help='only_1_fc layer in classifier')
    
    parser.add_argument('--resnet18', action='store_true', default=False, help='resnet18')
    parser.add_argument('--vgg', action='store_true', default=False, help='vgg')
    parser.add_argument('--wide', action='store_true', default=False, help='wider network')
    parser.add_argument('--sample', action='store_true', default=False, help='sample')
    parser.add_argument('--not-sample', action='store_true', default=False, help='not_sample')
    parser.add_argument('--frozen-model', action='store_true', default=False, help='not_sample')
    parser.add_argument('--dup-train', type=int, default=1, metavar='N', help='duplicate train batch')    
    parser.add_argument('--fix-fc1', action='store_true', default=False, help='fix-fc1')
    parser.add_argument('--retrain-last-layers', action='store_true', default=False, help='retrain-last-layers')    
    parser.add_argument('--warmup', action='store_true', default=False, help='warmup')    
    parser.add_argument('--warmup-l6', action='store_true', default=False, help='warmup')    
    parser.add_argument('--fix-fc1-revival', action='store_true', default=False, help='fix_fc1_revival')    
    parser.add_argument('--fc1-lr', type=float, default=1.0, metavar='N',help='learning rate (default: 1.0)')
    parser.add_argument('--mixup', action='store_true', default=False, help='mixup')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1)')   
    parser.add_argument('--add_no_mixup_acc', action='store_true', default=False, help='add_no_mixup_acc')
    parser.add_argument('--svm', action='store_true', default=False, help='svm loss')

    args = parser.parse_args()
    
    # TODO: chec no issue
    args.save_file = str('mnist' if args.mnist else 'cifar10') + args.suffix
    
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
    writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_ver2/' if args.ver2 else '/') + str(writer_suffix)
#     writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_new_run')
    writer = SummaryWriter(writer_name)
    
    # Data
    if args.cifar10 and not args.cifar100:
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
        trainloader_nomixup = torch.utils.data.DataLoader(trainset, **test_kwargs)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, **test_kwargs)

    elif args.cifar100 and args.cifar10:
        print('==> Preparing CIFAR100 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)

        testset = torchvision.datasets.CIFAR100(
            root='../data', train=False, download=True, transform=transform_test)
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
            if args.ver2:                
                net = FPNet_CIFAR10_ver2()
                if args.vgg:
                    print("Training VGG_SMALL for CIFAR10")
                    net = VGG_SMALL()                
            else:
                print("Training FP-Net for CIFAR10")
                net = FPNet_CIFAR10()
                if args.vgg:
                    net = VGG_SMALL()
        else:
            if args.ver2:
                print ("Training LR-Net for CIFAR10 | ver2")
#                 net = LRNet_CIFAR10_ver2(writer)
                if args.sample:
                    net = LRNet_CIFAR10_ver2_sample(writer, args)
                elif args.not_sample:
                    net = LRNet_CIFAR10_ver2_not_sample(writer, args)       
#                     net = LRNet_CIFAR10_ver2_l1(writer, args) #shooli                        
                else:
                    net = LRNet_CIFAR10_ver2X(writer, args)                    
#                 net = LRNet_CIFAR10_ver2XXX(writer)
            else:
                print ("Training LR-Net for CIFAR10")
                net = LRNet_CIFAR10()

            if args.load_pre_trained:
#                 if args.frozen_model:                   
#                     print("Loading Parameters for CIFAR10 | ver2")
#                     test_model = LRNet_CIFAR10_ver2_not_sample().to(device)
#                     test_model.load_state_dict(torch.load('saved_models/cifar10_lrnet_ver2_no_gumbel_29_11.pt'))

#                     test_model_fp = FPNet_CIFAR10().to(device)
#                     test_model_fp.load_state_dict(torch.load('saved_models/cifar10_fp.pt'))                    
                    
#                     state_dict = test_model.state_dict()
#                     state_dict_fp = test_model_fp.state_dict()
#                     with torch.no_grad():
#                         net.conv1.alpha.copy_(state_dict['conv1.alpha'])
#                         net.conv1.betta.copy_(state_dict['conv1.betta'])
#                         net.conv2.alpha.copy_(state_dict['conv2.alpha'])
#                         net.conv2.betta.copy_(state_dict['conv2.betta'])
#                         net.conv3.alpha.copy_(state_dict['conv3.alpha'])
#                         net.conv3.betta.copy_(state_dict['conv3.betta'])
#                         net.conv4.alpha.copy_(state_dict['conv4.alpha'])
#                         net.conv4.betta.copy_(state_dict['conv4.betta'])
#                         net.conv5.alpha.copy_(state_dict['conv5.alpha'])
#                         net.conv5.betta.copy_(state_dict['conv5.betta'])

#                         net.conv1.test_weight.copy_(state_dict['conv1.test_weight'])
#                         net.conv2.test_weight.copy_(state_dict['conv2.test_weight'])
#                         net.conv3.test_weight.copy_(state_dict['conv3.test_weight'])
#                         net.conv4.test_weight.copy_(state_dict['conv4.test_weight'])
#                         net.conv5.test_weight.copy_(state_dict['conv5.test_weight'])                        
                       
#                         net.conv1.bias.copy_(state_dict['conv1.bias'])
#                         net.conv2.bias.copy_(state_dict['conv2.bias'])
#                         net.conv3.bias.copy_(state_dict['conv3.bias'])
#                         net.conv4.bias.copy_(state_dict['conv4.bias'])
#                         net.conv5.bias.copy_(state_dict['conv5.bias'])

#                         alpha6, betta6 = find_sigm_weights(test_model_fp.conv6.weight, False)
#                         net.conv6.initialize_weights(alpha6, betta6)
#                         net.conv6.bias.copy_(state_dict_fp['conv6.bias'])
# #                         net.fc1.weight.copy_(state_dict['fc1.weight'])
# #                         net.fc1.bias.copy_(state_dict['fc1.bias'])
#                 else:
                print("Loading Parameters for CIFAR10")
                print("Loading model: saved_models/cifar10_fp.pt")
                test_model = FPNet_CIFAR10().to(device)
                test_model.load_state_dict(torch.load('saved_models/cifar10_fp.pt'))

                if args.wide:
                    print("Loading model: saved_models//cifar10_vgg_wide_5_11_fp.pt")
                    test_model = VGG_SMALL().to(device)                
                    test_model.load_state_dict(torch.load('saved_models//cifar10_vgg_wide_5_11_fp.pt'))
                elif args.vgg:
                    print("Loading model: saved_models/cifar10_vgg_small_sgd_fp.pt")
                    test_model = VGG_SMALL().to(device)                
#                     test_model.load_state_dict(torch.load('saved_models/cifar10_vgg_small_sgd_fp.pt'))
#                     test_model.load_state_dict(torch.load('saved_models/cifar10_vgg_fp_l6_is128_23_11_fp.pt'))
                    test_model.load_state_dict(torch.load('saved_models/cifar10_vgg_fp_l6_is256_23_11_fp.pt'))

                alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
                alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)
                alpha3, betta3 = find_sigm_weights(test_model.conv3.weight, False)
                alpha4, betta4 = find_sigm_weights(test_model.conv4.weight, False)
                alpha5, betta5 = find_sigm_weights(test_model.conv5.weight, False)
                alpha6, betta6 = find_sigm_weights(test_model.conv6.weight, False)

                # TODO shooli net.conv1.initialize_weights(alpha1, betta1)
                if net.conv1_is_normal is not True:
                    net.conv1.initialize_weights(alpha1, betta1)
                net.conv2.initialize_weights(alpha2, betta2)
                net.conv3.initialize_weights(alpha3, betta3)
                net.conv4.initialize_weights(alpha4, betta4)
                net.conv5.initialize_weights(alpha5, betta5)
                net.conv6.initialize_weights(alpha6, betta6)

                state_dict = test_model.state_dict()
                with torch.no_grad():
                    if net.conv1_is_normal is not True:
                        net.conv1.bias.copy_(state_dict['conv1.bias'])
                    net.conv2.bias.copy_(state_dict['conv2.bias'])
                    net.conv3.bias.copy_(state_dict['conv3.bias'])
                    net.conv4.bias.copy_(state_dict['conv4.bias'])
                    net.conv5.bias.copy_(state_dict['conv5.bias'])
                    net.conv6.bias.copy_(state_dict['conv6.bias'])
                    if args.vgg:
                        net.fc1.weight.copy_(state_dict['fc1.weight'])
                        net.fc1.bias.copy_(state_dict['fc1.bias'])                        
                    elif not args.only_1_fc: # TODO
                        net.fc1.weight.copy_(state_dict['fc1.weight'])
                        net.fc1.bias.copy_(state_dict['fc1.bias'])
                        net.fc2.weight.copy_(state_dict['fc2.weight'])
                        net.fc2.bias.copy_(state_dict['fc2.bias'])

                    # net.conv1.bias = normalize_layer(test_model.conv1.bias)
                    # net.conv2.bias = normalize_layer(test_model.conv2.bias)
                    # net.conv3.bias = normalize_layer(test_model.conv3.bias)
                    # net.conv4.bias = normalize_layer(test_model.conv4.bias)
                    # net.conv5.bias = normalize_layer(test_model.conv5.bias)
                    # net.conv6.bias = normalize_layer(test_model.conv6.bias)
                    #
                    # net.fc1.weight = test_model.fc1.weight
                    # net.fc1.bias = test_model.fc1.bias
                    # net.fc2.weight = test_model.fc2.weight
                        # net.fc2.bias = test_model.fc2.bias

    elif args.mnist:
        if args.full_prec:
            if args.ver2:
                print ("Training FP-Net for MNIST | ver2")
                net = FPNet_ver2().to(device)                       
            else:
                print ("Training FP-Net for MNIST")
                net = FPNet().to(device)
        else:
            if args.ver2:
                print ("Training LR-Net for MNIST | ver2")
#                 net = LRNet_ver2(writer).to(device)
                net = LRNet_ver2XXX(writer, args).to(device)

                if args.load_pre_trained:
                    print("Loading Parameters for MNIST | ver2")
                    test_model = FPNet_ver2().to(device)
                    test_model.load_state_dict(torch.load('saved_models/mnist_fp.pt'))
                    alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
                    alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)

                    net.conv1.initialize_weights(alpha1, betta1)
                    net.conv2.initialize_weights(alpha2, betta2)

                    state_dict = test_model.state_dict()
                    with torch.no_grad():
                        net.conv1.bias.copy_(state_dict['conv1.bias'])
                        net.conv2.bias.copy_(state_dict['conv2.bias'])
                        net.fc1.weight.copy_(state_dict['fc1.weight'])
                        net.fc1.bias.copy_(state_dict['fc1.bias'])
                        net.fc2.weight.copy_(state_dict['fc2.weight'])
                        net.fc2.bias.copy_(state_dict['fc2.bias'])                
                
            else:
                print ("Training LR-Net for MNIST")
                net = LRNet().to(device)

            if args.load_pre_trained:
                print("Loading Parameters for MNIST")
                test_model = FPNet().to(device)
                test_model.load_state_dict(torch.load('saved_models/mnist_fp.pt'))
                # test_model.eval()

                alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False, args.binary_mode)
                alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False, args.binary_mode)

                net.conv1.initialize_weights(alpha1, betta1)
                net.conv2.initialize_weights(alpha2, betta2)

                net.conv1.bias = test_model.conv1.bias
                net.conv2.bias = test_model.conv2.bias

                net.fc1.weight = test_model.fc1.weight
                net.fc1.bias = test_model.fc1.bias
                net.fc2.weight = test_model.fc2.weight
                net.fc2.bias = test_model.fc2.bias

    # if device == 'cuda':
    # # TODO    net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    if args.resnet18:                
        net = resnet18()    
        net = resnet20()    
        
    if device == 'cuda':
        if args.cudnn:
            print('==> Using cudnn.benchmark = True')
            cudnn.benchmark = True
        else:
            print('==> Using cudnn.benchmark = False && torch.backends.cudnn.deterministic = True')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('./checkpoint/ckpt.pth')
        # net.load_state_dict(checkpoint['net'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']
        # TODO
        net.load_state_dict(torch.load('../model_2/LRNet/saved_model/best_cifar10_cnn.pt'))

    layer1_decay = 10**(-20)
    bn_decay = 10**((-1)*args.bn_wd)
    weight_decay = 10**((-1)*args.wd)
    probability_decay = 10**((-1)*args.pd)
    criterion = nn.CrossEntropyLoss()

    if not args.sgd:
        if args.full_prec or args.resnet18:
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
        elif args.mnist:
            if args.ver2:
                # TODO
#                 optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
                optimizer = optim.Adam([
                    {'params': net.conv1.parameters(), 'weight_decay': probability_decay},
                    # {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                    {'params': net.fc1.parameters(), 'weight_decay': weight_decay},
                    {'params': net.fc2.parameters(), 'weight_decay': weight_decay},
                    
                    {'params': net.sign_prob1.parameters(), 'weight_decay': weight_decay},
                    {'params': net.sign_prob2.parameters(), 'weight_decay': weight_decay},
                    
#                     {'params': net.bn1.parameters()},
                    # {'params': net.bn2.parameters()},
                    {'params': net.bn3.parameters()}
                ], lr=args.lr, weight_decay=weight_decay)
            else:
                optimizer = optim.Adam([
                    {'params': net.conv1.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                    {'params': net.fc1.parameters(), 'weight_decay': weight_decay},
                    {'params': net.fc2.parameters(), 'weight_decay': weight_decay}
                    # {'params': net.bn1.parameters()},
                    # {'params': net.bn2.parameters()}
                ], lr=args.lr, weight_decay=weight_decay)
        elif args.cifar10:
            if args.ver2:
                print('==> Building CIFAR10 | ver2 optimizer')                
                # TODO
#                 optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
                if args.only_1_fc:
                    if args.not_sample:

#                         wd_decay = list()
#                         bias_decay = list()
#                         layer1_prob_decay = list()
#                         prob_decay = list()
#                         no_decay = list()                
#                         for name,param in net.named_parameters():
#                             if any(substring in str(name) for substring in ["conv1.alpha", "conv1.betta"]):
#                                 layer1_prob_decay.append(param)            
#                             elif any(substring in str(name) for substring in ["alpha", "betta"]):
#                                 prob_decay.append(param)            
#                             elif any(substring in str(name) for substring in ["gain", "weight"]):
#                                 wd_decay.append(param)            
#                             elif any(substring in str(name) for substring in ["bias"]):
#                                 bias_decay.append(param)            
#                             else:
#                                 no_decay.append(param)            

#                             optimizer = optim.Adam(
#                             [
#                                 {"params": layer1_prob_decay, "weight_decay": layer1_decay},
#                                 {"params": prob_decay, "weight_decay": probability_decay},
#                                 {"params": wd_decay,   "weight_decay": weight_decay},
#                                 {"params": bias_decay, "weight_decay": weight_decay},
#                                 {"params": no_decay,   "weight_decay": 0}                     
#                             ],
#                             args.lr, weight_decay=weight_decay)   

                        if net.conv1_is_normal:
                            layer1_decay = weight_decay
                        # here3
                        optimizer = optim.Adam([
#                                 {'params': net.conv1.parameters(), 'weight_decay': weight_decay, 'name': 'conv1'}, # TODO shooli     
#                                 {'params': net.bn1.parameters(), 'weight_decay': weight_decay, 'name': 'bn1'},
                            
                                {'params': net.conv1.parameters(), 'weight_decay': layer1_decay, 'name': 'conv1'}, # TODO shooli
                                {'params': net.conv2.parameters(), 'weight_decay': probability_decay, 'name': 'conv2'},
                                {'params': net.conv3.parameters(), 'weight_decay': probability_decay, 'name': 'conv3'},
                                {'params': net.conv4.parameters(), 'weight_decay': probability_decay, 'name': 'conv4'},
                                {'params': net.conv5.parameters(), 'weight_decay': probability_decay, 'name': 'conv5'},
                                {'params': net.conv6.parameters(), 'weight_decay': probability_decay, 'name': 'conv6'},
#                                     {'params': net.conv1.parameters(), 'name': 'conv1'}, # TODO shooli
#                                 {'params': net.conv2.parameters(), 'name': 'conv2'},
#                                 {'params': net.conv3.parameters(), 'name': 'conv3'},
#                                 {'params': net.conv4.parameters(), 'name': 'conv4'},
#                                 {'params': net.conv5.parameters(), 'name': 'conv5'},
#                                 {'params': net.conv6.parameters(), 'name': 'conv6'},

#                                     {'params': net.sign_prob1.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn1'},
#                                     {'params': net.sign_prob2.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn2'},
#                                     {'params': net.sign_prob3.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn3'},
#                                     {'params': net.sign_prob4.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn4'},
#                                     {'params': net.sign_prob5.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn5'},
                                {'params': net.sign_prob6.bn.parameters(), 'weight_decay': weight_decay, 'name': 'bn6'},

#                                 {'params': net.gain1.parameters(), 'weight_decay': weight_decay, 'name': 'gain1'},
#                                 {'params': net.gain2.parameters(), 'weight_decay': weight_decay, 'name': 'gain2'},
#                                 {'params': net.gain3.parameters(), 'weight_decay': weight_decay, 'name': 'gain3'},
#                                 {'params': net.gain4.parameters(), 'weight_decay': weight_decay, 'name': 'gain4'},
#                                 {'params': net.gain5.parameters(), 'weight_decay': weight_decay, 'name': 'gain5'},
#                                 {'params': net.gain6.parameters(), 'weight_decay': weight_decay, 'name': 'gain6'},

                                {'params': net.fc1.parameters(), 'name': 'fc1'}
#                                 {'params': [p for p in net.fc1.parameters() if p.requires_grad], 'name': 'fc1'}
                            ], lr=args.lr, weight_decay=weight_decay)                                                             
                    
                    else:
                        optimizer = optim.Adam([
                                {'params': net.conv1.parameters(), 'weight_decay': layer1_decay, 'name': 'conv1'},
                                {'params': net.conv2.parameters(), 'weight_decay': probability_decay, 'name': 'conv2'},
                                {'params': net.conv3.parameters(), 'weight_decay': probability_decay, 'name': 'conv3'},
                                {'params': net.conv4.parameters(), 'weight_decay': probability_decay, 'name': 'conv4'},
                                {'params': net.conv5.parameters(), 'weight_decay': probability_decay, 'name': 'conv5'},
                                {'params': net.conv6.parameters(), 'weight_decay': probability_decay, 'name': 'conv6'},

                                {'params': net.fc1.parameters(), 'name': 'fc1'},
                                {'params': net.bn6.parameters(), 'name': 'bn6'}
                            ], lr=args.lr, weight_decay=weight_decay)     
        
                else:
                    optimizer = optim.Adam([
                            {'params': net.conv1.parameters(), 'weight_decay': layer1_decay},
                            {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                            {'params': net.conv3.parameters(), 'weight_decay': probability_decay},
                            {'params': net.conv4.parameters(), 'weight_decay': probability_decay},
                            {'params': net.conv5.parameters(), 'weight_decay': probability_decay},
                            {'params': net.conv6.parameters(), 'weight_decay': probability_decay},
                            {'params': net.fc1.parameters()},
                            {'params': net.fc2.parameters()},
                            {'params': net.bn6.parameters()}
                        ], lr=args.lr, weight_decay=weight_decay)   
            else:
                print('==> Building CIFAR10 optimizer')
                parameters = list(net.named_parameters())
                # bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
                wght_params = [v for n, v in parameters if
                               ((("conv" in n) or ("fc" in n)) and ("weight" in n)) and v.requires_grad]
                prob_params = [v for n, v in parameters if (("alpha" in n) or ("betta" in n)) and v.requires_grad]
                bn_params   = [v for n, v in parameters if (("bn" in n) and ("weight" in n))]
                rest_params = [v for n, v in parameters if (("bias" in n))]                 
#                 rest_params = [v for n, v in parameters if
#                                ((("conv" in n) or ("fc" in n)) and ("bias" in n)) and v.requires_grad]
#                 rest_params = [v for n, v in parameters if (("bn" in n) or ("bias" in n))]  <- original
#                 rest_params = [v for n, v in parameters if (("bn" in n) and ("weight" in n))]
# conv
# alpha <- prob_params 
# betta <- prob_params 
# bias  <- rest_params
# BN
# weight <- bn_params
# bias   <- rest_params
# FC
# weight <- wght_params
# bias   <- rest_params

#                 wght_decay = dict()
#                 prob_decay = dict()    
#                 bias_decay = dict()    
#                 no_decay = dict()
#                 for name, m in net.named_parameters():
#                     print('checking {}'.format(name))
#                     if 'weight' in name:
#                         wght_decay[name] = m
#                     elif 'alpha' in name or 'betta' in name:
#                         prob_decay[name] = m
#                     elif 'bias' in name:
#                         bias_decay[name] = m
#                     else:
#                         no_decay[name] = m

#                 print(wght_decay.keys())
#                 print(prob_decay.keys())
#                 print(bias_decay.keys())
#                 print(no_decay.keys())

#                 optimizer = optim.Adam(
#                     [
#                         {"params": prob_decay.keys(), "weight_decay": probability_decay},
#                         {"params": wght_decay.keys(), "weight_decay": weight_decay},
#                         {"params": bias_decay.keys(), "weight_decay": bn_decay},
#                         {"params": no_decay,          "weight_decay": weight_decay}                     
#                     ],
#                     args.lr) 

                all_params = set(net.parameters())
                wd_decay = set()
                bias_decay = set()
                prob_decay = set()
                no_decay = set()
                for m in net.modules():
                    if isinstance(m, (lrnet_nn.LRnetConv2d)):
                        prob_decay.add(m.alpha)
                        prob_decay.add(m.betta)
                        bias_decay.add(m.bias)
                    elif isinstance(m, (nn.BatchNorm2d)):
#                         no_decay.add(m.weight)
#                         no_decay.add(m.bias)
                        wd_decay.add(m.weight)
                        bias_decay.add(m.bias)
                    elif isinstance(m, (nn.Linear)):
                        wd_decay.add(m.weight)
                        bias_decay.add(m.bias)
#                     else:
#                         no_decay.add(m.weight)
#                         no_decay.add(m.bias)  
                        
#                 print(wd_decay)
#                 print(prob_decay)
#                 print(bias_decay)
#                 print(no_decay)        

                optimizer = optim.Adam(
                    [
                        {"params": list(prob_decay), "weight_decay": probability_decay},
                        {"params": list(wd_decay),   "weight_decay": weight_decay},
                        {"params": list(bias_decay), "weight_decay": bn_decay},
                        {"params": list(no_decay),   "weight_decay": 0}                     
                    ],
                    args.lr)   

#                 optimizer = optim.Adam(
#                     [
#                         # {"params": bn_params, "weight_decay": 0 if args.no_bn_decay else args.weight_decay},
#                         {"params": prob_params, "weight_decay": probability_decay},
#                         {"params": wght_params, "weight_decay": weight_decay},
# #                         {"params": bn_params,   "weight_decay": bn_decay if args.add_bn_bias_decay else 0},
#                         {"params": bn_params,   "weight_decay": bn_decay},
#                         {"params": rest_params, "weight_decay": weight_decay},                        
#                     ],
#                     args.lr)

                # optimizer = optim.Adam(net.parameters(), lr=args.lr)
#                 optimizer = optim.Adam([
#                     {'params': net.conv1.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.conv3.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.conv4.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.conv5.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.conv6.parameters(), 'weight_decay': probability_decay},
#                     {'params': net.fc1.parameters(), 'weight_decay': weight_decay}, # TODO
#                     {'params': net.fc2.parameters(), 'weight_decay': weight_decay}, # TODO
#                     {'params': net.bn1.parameters()},
#                     {'params': net.bn2.parameters()},
#                     {'params': net.bn3.parameters()},
#                     {'params': net.bn4.parameters()},
#                     {'params': net.bn5.parameters()},
#                     {'params': net.bn6.parameters()}
#                 ], lr = args.lr, weight_decay = weight_decay)

        if args.annealing_sched:
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        if args.full_prec:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5 * weight_decay)
        elif args.mnist:
            optimizer = optim.SGD([
                {'params': net.conv1.parameters(), 'weight_decay': probability_decay},
                {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                {'params': net.fc1.parameters(), 'weight_decay': 5*weight_decay},
                {'params': net.fc2.parameters(), 'weight_decay': 5*weight_decay},
                {'params': net.bn1.parameters()},
                {'params': net.bn2.parameters()}
            ], lr=args.lr, momentum=0.9)
            # ], lr=args.lr, momentum=0.9, weight_decay=5*weight_decay)
        elif args.cifar10:
            if args.ver2:
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5 * weight_decay)
            else:
                optimizer = optim.SGD([
                    {'params': net.conv1.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv3.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv4.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv5.parameters(), 'weight_decay': probability_decay},
                    {'params': net.conv6.parameters(), 'weight_decay': probability_decay},
                    {'params': net.fc1.parameters()},
                    {'params': net.fc2.parameters()},
                    {'params': net.bn1.parameters()},
                    {'params': net.bn2.parameters()},
                    {'params': net.bn3.parameters()},
                    {'params': net.bn4.parameters()},
                    {'params': net.bn5.parameters()},
                    {'params': net.bn6.parameters()}
                ], lr=args.lr, momentum=0.9, weight_decay=5*weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    net = net.to(device)
    
    
#########################################

    if args.warmup:
        for g in optimizer.param_groups:
            if args.warmup_l6:
                if g['name'] == 'conv6':
                    g['lr'] = g['lr'] * 0.0001
            else:
                g['lr'] = g['lr'] * 0.01

    if args.fc1_lr != 1.0:
        for g in optimizer.param_groups:
            if g['name'] == 'fc1':
                g['lr'] = g['lr'] * args.fc1_lr
                
#########################################
    
    
    if args.save_file != 'no_need_to_save':
        file_name = "tmp_logs/" + str(args.save_file) + ".log"
        f = open(file_name, "w")
        print(args, file=f)
    else:
        print(args)
        f = None

    if args.ver2 and not args.full_prec:        
        if hasattr(net, 'sampled_last_layer'):
            print("sampled_last_layer:", net.sampled_last_layer)
        print("bn_s:", net.bn_s)
        print("gumbel:", net.gumbel)
        if hasattr(net, 'gumble_last_layer'):        
            print("gumble_last_layer:", net.gumble_last_layer)
        print("gain:", net.gain)
        if hasattr(net, 'only_1_fc'):                
            print("only_1_fc:", net.only_1_fc)   

        if hasattr(net, 'sampled_last_layer'):
            print("sampled_last_layer:", net.sampled_last_layer, file=f)
        print("bn_s:", net.bn_s, file=f)
        print("gumbel:", net.gumbel, file=f)
        if hasattr(net, 'gumble_last_layer'):                
            print("gumble_last_layer:", net.gumble_last_layer, file=f)
        print("gain:", net.gain, file=f)
        if hasattr(net, 'only_1_fc'):                        
            print("only_1_fc:", net.only_1_fc, file=f)              

# ########################################################################    
#     images, labels = next(iter(testloader))
#     images, labels = images.cuda(), labels.cuda()    
# #     example_data, example_target = examples.next()    
# #     example_data, example_target = examples.next()
# #     for i in range(0,6):
# #         plt.subplot(2,3,i+1)
# #         plt.imshow(example_data[i][0], cmap='gray')
# #     img_grid = torchvision.utils.make_grid(example_data)
# #     writer.add_image('my_images', img_grid)
# #     writer.add_graph(net, images)
# #########################################################################

    freeze_mask = np.full((args.lnum), 1).tolist()
    freeze_arr = np.arange(0, args.lnum) * args.step + args.start 
    entropy_level = 0.69

    cas = False
    if args.full_prec:
        warmup = False

    warmup = False 
    l6_conv = False
    
    if args.fix_fc1:
        net.fc1.weight.requires_grad = False
        net.fc1.bias.requires_grad = False        
        
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if args.freeze:
            net.train_mode_freeze(freeze_mask)
        else:
            if not args.full_prec:
                net.train_mode_switch()

        if args.ver2:
#             net.use_batch_stats_switch(True)
            net.tensorboard_train = True
            if args.freeze:
#                 freeze_arr = [1,2,3,4,5]  #0
#                 freeze_arr = [50,90,130,170,210]  #0
#                 freeze_arr = [100,130,170,210,240]  #1
#                 freeze_arr = [150,180,210,240,270]
#                 freeze_arr = [70,110,150,210,240]
#                 freeze_arr = [70,110,150,500,500]
#                 freeze_arr = [70,110,150,240,500]  #2 <- best
#                 freeze_arr = [70,110,150,210,240,270]
#                 freeze_arr = [40,100,160,240,500]  #3
                if epoch==100000:
                    net.unfreeze_all_layer()
                else:
                    warmup = False
                    
                    is_pool_layer = False
                    is_update_tau = False
                    is_gain = False
                    
#                     net.update_tau(net.sign_prob1, 0.1)                    

                    if epoch==freeze_arr[0]:
                        # TODO shooli
                        if net.conv1_is_normal is not True:
                            net.freeze_layer(net.conv1, net.sign_prob1, net.conv2, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain1)
#                         net.freeze_l1_layer(net.conv1, net.sign_prob1, net.conv2, args.trials, net, criterion, device, trainloader, args, f)
#                         net.freeze_layer(net.conv1, net.sign_prob1, net.conv2)
                            warmup = True
                            freeze_mask = [0,1,1,1,1,1]     
#                             net.update_tau(net.sign_prob2, 0.1)
                    if epoch==freeze_arr[1]:
                        if net.conv1_is_normal:
                            for param in net.conv1.parameters():
                                param.requires_grad = False  
                            for param in net.bn1.parameters():
                                param.requires_grad = False                              
                        net.freeze_layer(net.conv2, net.sign_prob2, net.conv3, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain2, pool_layer=None if is_pool_layer is False else net.pool2) # net.pool2  
                        freeze_mask = [0,0,1,1,1,1]
#                         net.update_tau(net.sign_prob3, 0.1)
                        warmup = True                       
                    if epoch==freeze_arr[2]:
                        net.freeze_layer(net.conv3, net.sign_prob3, net.conv4, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain3)   
                        freeze_mask = [0,0,0,1,1,1]
                        warmup = True
#                         net.update_tau(net.sign_prob4, 0.7)
#                         net.update_tau(net.sign_prob5, 0.7)
#                         net.update_tau(net.sign_prob6, 0.7)    
#                         net.update_tau(net.sign_prob4, 0.1)
                    if epoch==freeze_arr[3]:
                        net.freeze_layer(net.conv4, net.sign_prob4, net.conv5, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain4, pool_layer=None if is_pool_layer is False else net.pool4) # net.pool4   
                        freeze_mask = [0,0,0,0,1,1]
                        warmup = True
#                         net.update_tau(net.sign_prob5, 0.5) # 0.4
#                         net.update_tau(net.sign_prob6, 0.5) # 0.4
#                         net.update_tau(net.sign_prob5, 0.1)
                    if epoch==freeze_arr[4]:
                        net.freeze_layer(net.conv5, net.sign_prob5, net.conv6, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain5)
                        freeze_mask = [0,0,0,0,0,1]
#                         net.update_tau(net.sign_prob6, 0.1)
                        warmup = True
                    if epoch==freeze_arr[5]:
#                         if args.frozen_model:
                        if args.retrain_last_layers:
                            with torch.no_grad():
                                alpha6, betta6 = find_sigm_weights(test_model.conv6.weight, False)
                                net.conv6.initialize_weights(alpha6, betta6)
                                net.conv6.bias.copy_(state_dict['conv6.bias'])
                                net.fc1.reset_parameters()
                        else:                      
                            if net.sampled_last_layer or args.sample:
                                net.freeze_last_layer(net.conv6, args.trials, net, criterion, device, trainloader, args, f)
                            else:
                                net.freeze_layer(net.conv6, net.sign_prob6, net.fc1, args.trials, net, criterion, device, trainloader, args, f, update_tau=is_update_tau, gain_layer=None if is_gain is False else net.gain6, pool_layer=None if is_pool_layer is False else net.pool6) # net.pool6
                            freeze_mask = [0,0,0,0,0,0]
                            warmup = True
                            l6_conv = False
                            
                            if args.fix_fc1_revival:
                                if args.fc1_lr != 1.0:
                                    for g in optimizer.param_groups:
                                        if g['name'] == 'fc1':
                                            g['lr'] = args.lr                            
#                             net.fc1.weight.requires_grad = True
#                             net.fc1.bias.requires_grad = True                             
                        
#             if args.frozen_model and not args.freeze:                       
#                 net.just_freeze_layer(net.conv1, net.sign_prob1, net.conv2)
#                 net.just_freeze_layer(net.conv2, net.sign_prob2, net.conv3)
#                 net.just_freeze_layer(net.conv3, net.sign_prob3, net.conv4)
#                 net.just_freeze_layer(net.conv4, net.sign_prob4, net.conv5)
#                 net.just_freeze_layer(net.conv5, net.sign_prob5, net.conv6)
#                 if epoch==200:
#                     net.just_freeze_layer(net.conv6, net.sign_prob6, net.fc1)
        warmup = False

        train_acc, entropy_level = train(net, criterion, epoch, device, trainloader, optimizer, args, f, writer, warmup, entropy_level, l6_conv)

        if args.add_no_mixup_acc:        
            _a, _b, train_no_mixup_acc = test(net, criterion, epoch, device, trainloader_nomixup, args, 0, 0, test_mode=True, f=f, eval_mode=True, dont_save=True)
        
#         if epoch <= 200: # 79 # 100
#             warmup_factor = 1.027 # 1.04715
#             for g in optimizer.param_groups:
#                 g['lr'] = 0.00001 * (warmup_factor ** epoch)
#                 new_lr = g['lr']
#         else:
#             for g in optimizer.param_groups:
#                 g['lr'] = 0.00001 * 35000 * (warmup_factor ** (-epoch))
#                 new_lr = g['lr']                    

        for g in optimizer.param_groups:
            if g['name'] == 'fc1':
                fc1_lr = g['lr']
            if g['name'] == 'conv1':
                conv1_lr = g['lr']                
            if g['name'] == 'conv6':
                conv6_lr = g['lr']                                
        writer.add_scalars('lr', {'fc1_lr':fc1_lr,'conv1_lr':conv1_lr, 'conv6_lr':conv6_lr}, epoch)                
                
        if args.warmup:
            if epoch < 100: # 79 # 100
                warmup_factor = 1.04715 # 1.027 # 1.04715
                for g in optimizer.param_groups:
                    if args.warmup_l6:
                        if g['name'] == 'conv6':
                            g['lr'] = warmup_factor * g['lr']
#                             new_lr = g['lr']
#                             writer.add_scalar("learning_rate/conv_6_lr", new_lr, epoch)
#                         if g['name'] == 'conv1':
#                             writer.add_scalar("learning_rate/conv_1_lr", g['lr'], epoch)                            
                    else:
                        g['lr'] = warmup_factor * g['lr']
#                         new_lr = g['lr']       
#                         if g['name'] == 'conv1':
#                             writer.add_scalar("learning_rate/lr", g['lr'], epoch)      
        else:
            for g in optimizer.param_groups:
                new_lr = g['lr'] 
            writer.add_scalar("learning_rate/lr", new_lr, epoch)

        writer.add_scalar("acc/train", train_acc, epoch)
        if args.add_no_mixup_acc:
            writer.add_scalar("acc/train_no_mixup", train_no_mixup_acc, epoch)
            writer.add_scalars("accuracy", {'train_no_mixup': train_no_mixup_acc}, epoch)            
        writer.add_scalars("accuracy", {'train_cont': train_acc}, epoch)
        if args.ver2 and (args.full_prec is False):
            net.tensorboard_train = False
            print("iter is :", net.iteration_train)
            net.iteration_train = net.iteration_train + 1 
        best_acc, best_epoch, test_acc = test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch, test_mode=False, f=f, eval_mode=True, dont_save=True)  # note: model is saved only in test method below
        writer.add_scalar("cont acc/test", test_acc, epoch)
        writer.add_scalars("accuracy", {'test_cont': test_acc}, epoch)
        if args.sampled_test:
            if args.ver2:
                if args.freeze:
                    net.test_mode_freeze(freeze_mask)
                else:
                    net.test_mode_switch(1, 1)
#                 net.collect_stats_switch(True)
#                 _a, acc_norm, _b = test(net, criterion, epoch, device, testloader, args, 0, None, test_mode=True, f=f, eval_mode=True, dont_save=True)
#                 writer.add_scalar("acc_norm/test", acc_norm, epoch)
#                 net.collect_stats_switch(False)
                
                net.tensorboard_test = True
                best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader, args, best_sampled_acc, best_sampled_epoch, test_mode=False, f=f, eval_mode=True, dont_save=False, writer=writer)
                writer.add_scalar("sampled_acc/test", sampled_acc, epoch)
                writer.add_scalars("accuracy", {'test_sampled': sampled_acc}, epoch)
                net.tensorboard_test = False
                net.iteration_test = net.iteration_test + 1 
                
                if args.add_no_mixup_acc:
                    print_summary(train_no_mixup_acc, best_acc, best_sampled_acc, sampled_acc, f)
                else:
                    print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)
            else:
                for idx in range(0, args.options):
                    net.test_mode_switch(1, 1)
                    best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader, args, best_sampled_acc, best_sampled_epoch, test_mode=False, f=f, eval_mode=True, dont_save=False)
                    print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)
                writer.add_scalar("sampled_acc/test", sampled_acc, epoch)
                writer.add_scalars("accuracy", {'test_sampled': sampled_acc}, epoch)            

#         net.update_tau(net.sign_prob1, new_tau=0.0, tau_factor=0.9806, stop_epoch=None, stop_val=0.1) # 9815
#         net.update_tau(net.sign_prob2, new_tau=0.0, tau_factor=0.9838, stop_epoch=None, stop_val=0.1)
#         net.update_tau(net.sign_prob3, new_tau=0.0, tau_factor=0.9861, stop_epoch=None, stop_val=0.1)
#         net.update_tau(net.sign_prob4, new_tau=0.0, tau_factor=0.9878, stop_epoch=None, stop_val=0.1)
#         net.update_tau(net.sign_prob5, new_tau=0.0, tau_factor=0.9885, stop_epoch=None, stop_val=0.1)
#         net.update_tau(net.sign_prob6, new_tau=0.0, tau_factor=0.9897, stop_epoch=None, stop_val=0.1)
                
        scheduler.step()

    if args.ver2 or (args.save_file != 'no_need_to_save'):
        net = net.to('cpu')
        torch.save(net.state_dict(), "saved_models/" + str(args.save_file) + "_cpu.pt")
#             torch.save(net.state_dict(), "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + str(args.suffix) + ".pt")        
    if args.full_prec:
        torch.save(net.state_dict(), "saved_models/" + str(args.save_file) + "_fp.pt")


    writer.flush()
    writer.close()

    if args.save_file != 'no_need_to_save':
        f.close()

if __name__ == '__main__':
    main_train()

