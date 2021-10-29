import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np    
import string
import random


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     weight_decay = 10**((-1)*args.wd) # 1e-4
#     probability_decay = 10**((-1)*args.pd) # 1e-11
#     print("weight_decay: " + str(weight_decay))
#     print("probability_decay: " + str(probability_decay))
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         if args.cifar10:
#             if args.full_prec:
#                 loss = F.cross_entropy(output, target)
#                 ce_loss = loss
#             else:
#                 ce_loss = F.cross_entropy(output, target)
#                 loss = ce_loss + probability_decay * (torch.norm(model.conv1.alpha, 2) + torch.norm(model.conv1.betta, 2)
#                                                  + torch.norm(model.conv2.alpha, 2) + torch.norm(model.conv2.betta, 2)
#                                                  + torch.norm(model.conv3.alpha, 2) + torch.norm(model.conv3.betta, 2)
#                                                  + torch.norm(model.conv4.alpha, 2) + torch.norm(model.conv4.betta, 2)
#                                                  + torch.norm(model.conv5.alpha, 2) + torch.norm(model.conv5.betta, 2)
#                                                  + torch.norm(model.conv6.alpha, 2) + torch.norm(model.conv6.betta, 2)) \
#                        + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
#         else:
#             if args.full_prec:
#                 loss = F.cross_entropy(output, target)
#             else:
#                 loss = F.cross_entropy(output, target) + probability_decay * (torch.norm(model.conv1.alpha, 2)
#                                                                + torch.norm(model.conv1.betta, 2)
#                                                                + torch.norm(model.conv2.alpha, 2)
#                                                                + torch.norm(model.conv2.betta, 2)) + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
#         # optimizer.zero_grad()
#         if args.debug_mode:
#             torch.autograd.set_detect_anomaly(True)
#             loss.backward(retain_graph=True)
#         else:
#             loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tloss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), ce_loss.item(), loss.item()))
#
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\n' + str(tstring) +' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#     return (100. * correct / len(test_loader.dataset))

# Training
def train(net, criterion, epoch, device, trainloader, optimizer, args, f=None, writer=None, warmup=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    weight_decay = 10**((-1)*args.wd)
    probability_decay = 10**((-1)*args.pd)
    t0 = time.time()
    
    for g in optimizer.param_groups:
        curre_lr = g['lr']
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print("batch_idx: " + str(batch_idx))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if warmup:            
            warmup_factor = (batch_idx / len(trainloader)) * (batch_idx / len(trainloader))
            for g in optimizer.param_groups:
                g['lr'] = warmup_factor * curre_lr
                print ("g:", g['lr'])
            print ("warmup_factor:", warmup_factor)
    
#         if args.freeze:      
#             if epoch>=1000:
#                 net.unfreeze_all_layer()
#             else:            
# #                 freeze_arr = [50,90,130,170,210]
# #                 freeze_arr = [100,130,170,210,240]
# #                 freeze_arr = [150,180,210,240,270]   
# #                 freeze_arr = [70,110,150,210,240]
# #                 freeze_arr = [70,110,150,500,500]    
# #                 freeze_arr = [70,110,150,240,500]
#                 freeze_arr = [70,110,150,210,240, 270]
# #                 freeze_arr = [40,100,160,240,500]
#                 if epoch>=freeze_arr[0]:
#                     net.freeze_layer(net.conv1, net.sign_prob1, net.conv2)
#                 if epoch>=freeze_arr[1]:
#                     net.freeze_layer(net.conv2, net.sign_prob2, net.conv3)   
#                 if epoch>=freeze_arr[2]:
#                     net.freeze_layer(net.conv3, net.sign_prob3, net.conv4)   
#                 if epoch>=freeze_arr[3]:
#                     net.freeze_layer(net.conv4, net.sign_prob4, net.conv5)   
#                 if epoch>=freeze_arr[4]:
#                     net.freeze_layer(net.conv5, net.sign_prob5, net.conv6)           
#                 if epoch>=freeze_arr[5]:
#                     net.freeze_last_layer(net.conv6)                     
        
        outputs = net(inputs)
        # print("output: " + str(outputs))
        # print("targets: " + str(targets))

        loss = criterion(outputs, targets)
        # loss = criterion(outputs, targets) + probability_decay * (torch.norm(net.conv1.alpha, 2) + torch.norm(net.conv1.betta, 2)
        #                                          + torch.norm(net.conv2.alpha, 2) + torch.norm(net.conv2.betta, 2)
        #                                          + torch.norm(net.conv3.alpha, 2) + torch.norm(net.conv3.betta, 2)
        #                                          + torch.norm(net.conv4.alpha, 2) + torch.norm(net.conv4.betta, 2)
        #                                          + torch.norm(net.conv5.alpha, 2) + torch.norm(net.conv5.betta, 2)
        #                                          + torch.norm(net.conv6.alpha, 2) + torch.norm(net.conv6.betta, 2)) \
        #                                          + weight_decay * (torch.norm(net.fc1.weight, 2) + (torch.norm(net.fc2.weight, 2)))

        assert not torch.isnan(loss).any(), "loss nan before backward"
        assert not torch.isinf(loss).any(), "loss inf before backward"        
        if torch.isnan(loss).any():
            print("loss isnan: " + str(torch.isnan(loss).any()))
            exit(1)
        # print("loss1: " + str(loss))
        if args.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        assert not torch.isnan(loss).any(), "loss nan after backward"
        assert not torch.isinf(loss).any(), "loss inf after backward"              
        if args.debug:
            print("was backward")
        optimizer.step()
        if args.debug:
            print("was optimizer step")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.nohup:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tacc: {:.3f} \tloss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), 100.*correct/total, loss.item()))
                if f is not None:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tacc: {:.3f} \tloss: {:.6f}'.format(
                        epoch, batch_idx * len(inputs), len(trainloader.dataset),
                               100. * batch_idx / len(trainloader), 100. * correct / total, loss.item()), file=f)
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # if args.nohup:
        #     print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #     if f is not None:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\tloss: {:.6f}'.format(
        #             epoch, batch_idx * len(trainloader), len(trainloader.dataset),
        #                    100. * batch_idx / len(trainloader), loss.item(), loss.item()), file=f)
        # else:
        #     progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                  % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if warmup:
        for g in optimizer.param_groups:
            g['lr'] = curre_lr
            print ("g:", g['lr'])
        warmup = False
    writer.add_scalar("Loss/train", loss, epoch)
    print('{} seconds'.format(time.time() - t0))
    return (100.*correct/total)


def test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch, test_mode=False, f=None, eval_mode=True, dont_save=True, writer=None):
    # global best_acc
    if eval_mode:
        net.eval()
    else:
        net.train()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.nohup:
                print('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
                    test_loss, correct, total, 100. * correct / total))
                if f is not None:
                    print('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
                        test_loss, correct, total, 100. * correct / total), file=f)
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # if args.nohup:
            #     print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #     if f is not None:
            #         print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #               % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total), file=f)
            # else:
            #     progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if (acc > best_acc) and not test_mode:
        if (not args.dont_save) and (not dont_save):
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            dataset_name = 'mnist' if args.mnist else 'cifar10'
            net_type = '_fp' if args.full_prec else '_lrnet'
            isBinary = '_binary' if args.binary_mode else ''
            isVer2 = '_ver2' if args.ver2 else ''
            torch.save(net.state_dict(), "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + str(args.suffix) + ".pt")
        best_acc = acc
        best_epoch = epoch
        if writer is not None:
            writer.add_scalar("Loss/test", loss, epoch)
    if test_mode:
        best_acc = acc

    print("--> best accuracy is: " + str(best_acc) + " (epoch: " + str(best_epoch) + ")")
    if f is not None:
        print("--> best accuracy is: " + str(best_acc) + " (epoch: " + str(best_epoch) + ")", file=f)
    return best_acc, best_epoch, acc


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def find_weights(w, my_prints=False):
    if my_prints:
        print("w: " + str(w))
        print(w.size())
        print(type(w))
    # note: e^alpha + e^betta + e^gamma = 1
    p_max = 0.95
    p_min = 0.05
    w_norm = w / torch.std(w)
    e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
    if my_prints:
        print("e_alpha: " + str(e_alpha))
    e_alpha = torch.clamp(e_alpha, p_min, p_max)
    if my_prints:
        print("e_alpha.clip: " + str(e_alpha))
        print("e_alpha.size: " + str(e_alpha.size()))

    # betta = 0.5 * (1 + (w_norm / (1 - alpha)))
    e_betta = 0.5 * (w_norm - e_alpha + 1)
    if my_prints:
        print("e_betta: " + str(e_betta))
    e_betta = torch.clamp(e_betta, p_min, p_max)
    if my_prints:
        print("e_betta.clip: " + str(e_betta))

    alpha_prob = torch.log(e_alpha)
    betta_prob = torch.log(e_betta)
    gamma_prob = torch.log(torch.clamp((1 - e_alpha - e_betta), p_min, p_max))
    if my_prints:
        print("alpha_prob: " + str(alpha_prob))
        print("betta_prob: " + str(betta_prob))
        print("gamma_prob: " + str(gamma_prob))
    alpha_prob = alpha_prob.detach().cpu().clone().numpy()
    betta_prob = betta_prob.detach().cpu().clone().numpy()
    gamma_prob = gamma_prob.detach().cpu().clone().numpy()
    alpha_prob = np.expand_dims(alpha_prob, axis=-1)
    betta_prob = np.expand_dims(betta_prob, axis=-1)
    gamma_prob = np.expand_dims(gamma_prob, axis=-1)
    theta = np.concatenate((alpha_prob, betta_prob, gamma_prob), axis=4)
    if my_prints:
        print("theta: " + str(theta))
        print("theta.shape: " + str(np.shape(theta)))
    return theta


def find_sigm_weights(w, my_prints=False, binary_mode=False):
    if my_prints:
        print("w: " + str(w))
        print(w.size())
        print(type(w))

    p_max = 0.95
    p_min = 0.05
    w_norm = w / torch.std(w)
    e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
    e_betta = 0.5 * (1 + (w_norm / (1 - e_alpha)))

    if binary_mode:
        e_alpha = torch.zeros(w.size())
    else:
        e_alpha = torch.clamp(e_alpha, p_min, p_max)

    if my_prints:
        print("alpha.clip: " + str(e_alpha))
        print("alpha.size: " + str(e_alpha.size()))

    if my_prints:
        print("e_betta: " + str(e_betta))
    e_betta = torch.clamp(e_betta, p_min, p_max)
    if my_prints:
        print("e_betta.clip: " + str(e_betta))

    alpha_prob = torch.log(e_alpha / (1 - e_alpha))
    betta_prob = torch.log(e_betta / (1 - e_betta))
    if my_prints:
        print("alpha_prob: " + str(alpha_prob))
        print("betta_prob: " + str(betta_prob))
    alpha_prob = alpha_prob.detach().cpu().clone().numpy()
    betta_prob = betta_prob.detach().cpu().clone().numpy()
    alpha_prob = np.expand_dims(alpha_prob, axis=-1)
    betta_prob = np.expand_dims(betta_prob, axis=-1)
    return alpha_prob, betta_prob

def print_full_tensor(input, input_name):
    for i, val1 in enumerate(input):
        for j, val2 in enumerate(val1):
            for m, val3 in enumerate(val2):
#                 if (i == 157) and (j == 43) and (m == 5):
                if ((val3 > 1.0).any()):
                    print (str(input_name) + "(" + str(i) + ", " + str(j) + ", " + str(m) + "): " + str(val3))

def print_fullllll_tensor(input, input_name):
    for i, val1 in enumerate(input):
        for j, val2 in enumerate(val1):
                print (str(input_name) + "(" + str(i) + ", " + str(j) + ": " + str(val2))
            # for m, val3 in enumerate(val2):
            #     print (str(input_name) + "(" + str(i) + ", " + str(j) + ", " + str(m) + ": " + str(val3))

def print_neg_val(input, input_name):
    for i, val1 in enumerate(input):
        for j, val2 in enumerate(val1):
            for m, val3 in enumerate(val2):
                if (val3 < 0).any():
                    print (str(input_name) + "(" + str(i) + ", " + str(j) + ", " + str(m) + ": " + str(val3))

def print_summary(train_acc, best_acc, best_sampled_acc, t_sampled_acc, f):
    print("#################################")
    print('train_acc:       \t{:.3f}'.format(train_acc))
    print('best_acc:        \t{:.3f}'.format(best_acc))
    print('best_sampled_acc:\t{:.3f}'.format(best_sampled_acc))
    print('curr_sampled_acc:\t{:.3f}'.format(t_sampled_acc))
    print("#################################")
    if f is not None:
        print("#################################", file=f)
        print('train_acc:       \t{:.3f}'.format(train_acc), file=f)
        print('best_acc:        \t{:.3f}'.format(best_acc), file=f)
        print('best_sampled_acc:\t{:.3f}'.format(best_sampled_acc), file=f)
        print('curr_sampled_acc:\t{:.3f}'.format(t_sampled_acc), file=f)
        print("#################################", file=f)


def copy_net2net(net_s, net):
    params1 = net.named_parameters()
    params2 = net_s.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if (name1 in dict_params2) and any(x in name1 for x in ('conv', 'fc')):
            # print("dict_params2[" + str(name1) + "].data.copy_(param1.data)")
            dict_params2[name1].data.copy_(param1.data)

def mean_over_channel(input):
    batch_size = input.size(0)
    input1 = input.view(input.size(0), input.size(1), -1).mean(2).sum(0)
    input1 = input1 / batch_size
    # print("input1: \n" + str(input1))
    # print("input1 size: \n" + str(input1.size()))
    # print("#################################")
    mean = input1.repeat(input.size(0), 1).view(input.size(0), input.size(1), 1, 1)
    print(mean)
    return mean


# def mean_over_channel(input):
#     batch = input.view(input.size(0), input.size(1), -1)
#     nsize = batch.size(0)
#     mean = batch.mean(2).sum(0)
#     mean /= nsize
#     return mean

def assertnan(x, name):
    if torch.isnan(x).any():
        print(str(name) + " isnan: " + str(torch.isnan(x).any()))
        exit(1)
        
def collect_hist(desired_itr, iteration, writer, x, x_name, test_weight, alpha, betta):        
    if (iteration == desired_itr) and (writer is not None):
        if x is not None:            
            hist_name = str(x_name) + str(' x distribution ') + str(iteration)
            writer.add_histogram(hist_name, x, 0, bins='auto')
#         if test_weight is not None:
#             hist_name = str(x_name) + str(' test_weight distribution ') + str(iteration)
#             writer.add_histogram(hist_name, test_weight, 0)
#         if alpha is not None:
#             hist_name = str(x_name) + str(' alpha distribution ') + str(iteration)
#             writer.add_histogram(hist_name, alpha, 0)
#         if betta is not None:
#             hist_name = str(x_name) + str(' betta distribution ') + str(iteration)
#             writer.add_histogram(hist_name, betta, 0)  
 
def collect_m_v(writer, name, x, iteration): 
    if writer is not None:
        m,v = x
        writer.add_scalar("m" + str(name) + " mean", torch.mean(m), iteration)
        writer.add_scalar("m" + str(name) + " std", torch.std(m), iteration)
#         writer.add_scalar("m" + str(name) + " max", torch.max(m), iteration)
#         writer.add_scalar("m" + str(name) + " min", torch.min(m), iteration)
        writer.add_scalar("v" + str(name) + " mean", torch.mean(v), iteration)
        writer.add_scalar("v" + str(name) + " std", torch.std(v), iteration)
#         writer.add_scalar("v" + str(name) + " max", torch.max(m), iteration)
#         writer.add_scalar("v" + str(name) + " min", torch.min(m), iteration)                
#         self.writer.add_scalar("max(m" + str(name) + "-v" + str(name) + ")", torch.max(m/v), self.iteration)
#         self.writer.add_scalar("min(m" + str(name) + "-v" + str(name) + ")", torch.min(m/v), self.iteration)   
#         self.writer.add_scalar("mean(m" + str(name) + "-v" + str(name) + ")", torch.mean(m/v), self.iteration)
#         self.writer.add_scalar("std(m" + str(name) + "-v" + str(name) + ")", torch.std(m/v), self.iteration)
 
    
def take_silce(name, x):     
    m, v = x
#     m_slice = m[:,:,0:3,0:3]
#     v_slice = v[:,:,0:3,0:3]
    m_slice = m[:,:,:,:]
    v_slice = v[:,:,:,:]    
    with open('layers/' + str(name) + '_m.npy', 'wb') as f:
        m_slice = m_slice.data.cpu().numpy()
        np.save(f, m_slice)
    with open('layers/' + str(name) + '_v.npy', 'wb') as f:
        v_slice = v_slice.data.cpu().numpy()
        np.save(f, v_slice)      
        
        
def output_hist(name, x, net_input, layer, writer, N, iteration=None):             
    iteration_idx = str(iteration) if iteration is not None else ''
    for i in range(N):
        if i == 0:
            y = x[0,0,0,0].view(1)
        else:
            layer.test_mode_switch(1, 1)
            z = layer(net_input)
            y = torch.cat((y,z[0,0,0,0].view(1)))
    writer.add_histogram(str(name) + " [0,0,0,0] " + iteration_idx + " distribution", y, 0, bins='auto')         
    
    
def layer_hist(name, x, writer, iteration=None):             
    iteration_idx = str(iteration) if iteration is not None else ''
    writer.add_histogram(str(name) + " " + str(iteration_idx) + " distribution", x, 0, bins='auto')  

def id_generator(size=16, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

    
