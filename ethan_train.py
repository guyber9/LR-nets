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

from ethan_models import *
from utils import find_sigm_weights, train, test, print_summary, copy_net2net

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def main_train():
    parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
    parser.add_argument('--mnist', action='store_true', default=False, help='mnist flag')
    parser.add_argument('--cifar10', action='store_true', default=False, help='cifar10 flag')
    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=170, metavar='M',help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
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
    parser.add_argument('--nohup', action='store_true', default=True, help='nohup mode')
    parser.add_argument('--dont-save', action='store_true', default=False, help='dont_save mode')

    parser.add_argument('--sgd', action='store_true', default=False, help='run with sgd')

    parser.add_argument('--save-file', action='store', default='no_need_to_save', help='name of saved model')

    parser.add_argument('--debug', action='store_true', default=False, help='run with adam')

    parser.add_argument('--options', type=int, default=1, metavar='N', help='num_of_options for rand')
    parser.add_argument('--tickets', type=int, default=1, metavar='N', help='num of tickets')
    parser.add_argument('--sampled-test', action='store_true', default=True, help='sampled validation in training')

    parser.add_argument('--add-bn-bias-decay', action='store_true', default=False, help='dd weight decay to bn(weight/bias) and bias')

    parser.add_argument('--ver2', action='store_true', default=True, help='discretization for layer output')
    parser.add_argument('--cudnn', action='store_true', default=True, help='using cudnn benchmark=True')
    parser.add_argument('--collect_stats', action='store_true', default=False, help='collect_stats for test')

    parser.add_argument('--suffix', action='store', default='', help='suffix for saved model name')
    parser.add_argument('--annealing-sched', action='store_true', default=False, help='using CosineAnnealingLR')
    parser.add_argument('--freeze', action='store_true', default=False, help='freeze layers')

    parser.add_argument('--lnum', type=int, default=6, metavar='N', help='num of layers')
    parser.add_argument('--step', type=int, default=40, metavar='N', help='step size in freezeing')
    parser.add_argument('--start', type=int, default=200, metavar='N', help='starting point in freezeing')
    parser.add_argument('--trials', type=int, default=30, metavar='N', help='num of trials in freezeing')

    parser.add_argument('--only-1-fc', action='store_true', default=True, help='only_1_fc layer in classifier')
    
    parser.add_argument('--resnet18', action='store_true', default=False, help='resnet18')
    parser.add_argument('--vgg', action='store_true', default=False, help='vgg')
    parser.add_argument('--wide', action='store_true', default=False, help='wider network')
    parser.add_argument('--sample', action='store_true', default=False, help='sample')
    parser.add_argument('--not_sample', action='store_true', default=True, help='not_sample')
    parser.add_argument('--writer', action='store_true', default=False, help='not_sample')

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
    writer_name = "runs/" + str('mnist' if args.mnist else 'cifar10') + str('_ver2/' if args.ver2 else '/') + str(writer_suffix)
    
    if args.writer is True:
        writer = SummaryWriter(writer_name)
    else:
        writer = None
    
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
        if args.full_prec:
            print("Training FP-Net for CIFAR10 | ver2")            
            net = FPNet_CIFAR10_ver2()
        else:
            print ("Training LR-Net for CIFAR10 | ver2")
            net = LRNet_CIFAR10_act(writer, args)       

            if args.load_pre_trained:
                print("Loading Parameters for CIFAR10")
                print("Loading model: saved_models/cifar10_fp.pt")
                test_model = FPNet_CIFAR10().to(device)
                test_model.load_state_dict(torch.load('saved_models/cifar10_fp.pt'))
     
                alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
                alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)
                alpha3, betta3 = find_sigm_weights(test_model.conv3.weight, False)
                alpha4, betta4 = find_sigm_weights(test_model.conv4.weight, False)
                alpha5, betta5 = find_sigm_weights(test_model.conv5.weight, False)
                alpha6, betta6 = find_sigm_weights(test_model.conv6.weight, False)

                net.conv1.initialize_weights(alpha1, betta1)
                net.conv2.initialize_weights(alpha2, betta2)
                net.conv3.initialize_weights(alpha3, betta3)
                net.conv4.initialize_weights(alpha4, betta4)
                net.conv5.initialize_weights(alpha5, betta5)
                net.conv6.initialize_weights(alpha6, betta6)

                state_dict = test_model.state_dict()
                with torch.no_grad():
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

#                     net.fc1.weight = test_model.fc1.weight
#                     net.fc1.bias = test_model.fc1.bias

    if device == 'cuda':
        print('==> Using cudnn.benchmark = True')
        cudnn.benchmark = True


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        net.load_state_dict(torch.load('../model_2/LRNet/saved_model/best_cifar10_cnn.pt'))

    layer1_decay = 10**(-20)
    bn_decay = 10**((-1)*args.bn_wd)
    weight_decay = 10**((-1)*args.wd)
    probability_decay = 10**((-1)*args.pd)
    criterion = nn.CrossEntropyLoss()

    if args.full_prec or args.resnet18:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:                                                                                
        optimizer = optim.Adam([
                {'params': net.conv1.parameters(), 'weight_decay': layer1_decay}, # TODO shooli
                {'params': net.conv2.parameters(), 'weight_decay': probability_decay},
                {'params': net.conv3.parameters(), 'weight_decay': probability_decay},
                {'params': net.conv4.parameters(), 'weight_decay': probability_decay},
                {'params': net.conv5.parameters(), 'weight_decay': probability_decay},
                {'params': net.conv6.parameters(), 'weight_decay': probability_decay},
                {'params': net.fc1.parameters()}
            ], lr=args.lr, weight_decay=weight_decay)   

    if args.annealing_sched:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    net = net.to(device)
    
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

    freeze_mask = np.full((args.lnum), 1).tolist()
    freeze_arr = np.arange(0, args.lnum) * args.step + args.start 
    entropy_level = 0.69

    cas = False
    warmup = False
       
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if args.freeze:
            net.train_mode_freeze(freeze_mask)
        else:
            if not args.full_prec:
                net.train_mode_switch()

        if args.ver2:
            net.tensorboard_train = True
            if args.freeze:
                if epoch==100000:
                    net.unfreeze_all_layer()
                else:
                    warmup = False
                    if epoch==freeze_arr[0]:
                        net.freeze_layer(net.conv1, net.sign_prob1, net.conv2, args.trials, net, criterion, device, trainloader, args, f)
                        warmup = True
                        freeze_mask = [0,1,1,1,1,1]
                    if epoch==freeze_arr[1]:
                        net.freeze_layer(net.conv2, net.sign_prob2, net.conv3, args.trials, net, criterion, device, trainloader, args, f, None) # net.pool2  
                        freeze_mask = [0,0,1,1,1,1]
                        warmup = True                        
                    if epoch==freeze_arr[2]:
                        net.freeze_layer(net.conv3, net.sign_prob3, net.conv4, args.trials, net, criterion, device, trainloader, args, f)   
                        freeze_mask = [0,0,0,1,1,1]
                        warmup = True
                    if epoch==freeze_arr[3]:
                        net.freeze_layer(net.conv4, net.sign_prob4, net.conv5, args.trials, net, criterion, device, trainloader, args, f, None) # net.pool4   
                        freeze_mask = [0,0,0,0,1,1]
                        warmup = True
                    if epoch==freeze_arr[4]:
                        net.freeze_layer(net.conv5, net.sign_prob5, net.conv6, args.trials, net, criterion, device, trainloader, args, f)
                        freeze_mask = [0,0,0,0,0,1]
                        warmup = True
                    if epoch==freeze_arr[5]:
                        if net.sampled_last_layer or args.sample:
                            net.freeze_last_layer(net.conv6, args.trials, net, criterion, device, trainloader, args, f)
                        else:
                            net.freeze_layer(net.conv6, net.sign_prob6, net.fc1, args.trials, net, criterion, device, trainloader, args, f, None) # net.pool6
                        freeze_mask = [0,0,0,0,0,0]
                        warmup = True
                        
        train_acc, entropy_level = train(net, criterion, epoch, device, trainloader, optimizer, args, f, writer, warmup, entropy_level)

        for g in optimizer.param_groups:
            new_lr = g['lr']
        if writer is not None:
            writer.add_scalar("stats/lr", new_lr, epoch)

        if writer is not None:
            writer.add_scalar("acc/train", train_acc, epoch)
        if args.ver2:
            net.tensorboard_train = False
            print("iter is :", net.iteration_train)
            net.iteration_train = net.iteration_train + 1 
        best_acc, best_epoch, test_acc = test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch, test_mode=False, f=f, eval_mode=True, dont_save=True)  # note: model is saved only in test method below
        if writer is not None:
            writer.add_scalar("cont acc/test", test_acc, epoch)
        if args.sampled_test:
            if args.ver2:
                if args.freeze:
                    net.test_mode_freeze(freeze_mask)
                else:
                    net.test_mode_switch(1, 1)                
                net.tensorboard_test = True
                best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader, args, best_sampled_acc, best_sampled_epoch, test_mode=False, f=f, eval_mode=True, dont_save=False, writer=writer)
                if writer is not None:                
                    writer.add_scalar("sampled_acc/test", sampled_acc, epoch)
                net.tensorboard_test = False
                net.iteration_test = net.iteration_test + 1 

                print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)
            else:
                for idx in range(0, args.options):
                    net.test_mode_switch(1, 1)
                    best_sampled_acc, best_sampled_epoch, sampled_acc = test(net, criterion, epoch, device, testloader, args, best_sampled_acc, best_sampled_epoch, test_mode=False, f=f, eval_mode=True, dont_save=False)
                    print_summary(train_acc, best_acc, best_sampled_acc, sampled_acc, f)
                if writer is not None:                
                    writer.add_scalar("sampled_acc/test", sampled_acc, epoch)

        scheduler.step()

    if args.ver2 or (args.save_file != 'no_need_to_save'):
        net = net.to('cpu')
        torch.save(net.state_dict(), "saved_models/" + str(args.save_file) + "_cpu.pt")
#             torch.save(net.state_dict(), "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + str(args.suffix) + ".pt")        
    if args.full_prec:
        torch.save(net.state_dict(), "saved_models/" + str(args.save_file) + "_fp.pt")

    if writer is not None:                
        writer.flush()
        writer.close()

    if args.save_file != 'no_need_to_save':
        f.close()

if __name__ == '__main__':
    main_train()

