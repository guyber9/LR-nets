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

# Training
def train(net, criterion, epoch, device, trainloader, optimizer, args, f=None, writer=None, warmup=False, entropy_level=None):
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
        
    epoch_level = 200
        
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

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        entropy_level = None

#         if net.sign_prob6.test_forward:
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#         else:
#             outputs, p6 = net(inputs)
#             loss = criterion(outputs, targets) + calc_act_entropy(p6, 0.3, avg=True)             

#         if net.sign_prob6.test_forward and net.sign_prob5.test_forward:
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)    
#         elif net.sign_prob5.test_forward and not net.sign_prob6.test_forward:
#             outputs, p6 = net(inputs)       
#             loss = criterion(outputs, targets) + calc_act_entropy(p6, 0.3, avg=True)
#         else:
#             outputs, p6, p5 = net(inputs)
#             loss = criterion(outputs, targets) + calc_act_entropy(p6, 0.3, avg=True) + calc_act_entropy(p5, 0.3, avg=True)

#         if net.sign_prob6.test_forward:
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)                
#         else:
#             outputs, p6 = net(inputs)
#             if batch_idx == 0:
#                 if epoch == (epoch_level+1):
#                     entropy_level = calc_average_entropy(p6)
#                     entropy_level = entropy_level.item()
#                 elif epoch > (epoch_level+1):
#                     entropy_level = entropy_level * 0.994
#                 writer.add_scalar("entropy_level", entropy_level, epoch)
                
#             if epoch > epoch_level:
#                 loss = criterion(outputs, targets) + calc_act_entropy(p6, entropy_level, avg=False)                
#             else:
#                 loss = criterion(outputs, targets)
        
    
#         loss = criterion(outputs, targets) + calc_weights_entropy(net.conv6, 0.5) + calc_weights_entropy(net.conv5, 0.5) + calc_weights_entropy(net.conv4, 0.5)
#         loss = criterion(outputs, targets) + calc_weights_entropy(net.conv6, 0.6) + calc_weights_entropy(net.conv5, 0.6) # + calc_weights_entropy(net.conv4, 0.6) # + calc_weights_entropy(net.conv3, 0.6) + calc_weights_entropy(net.conv2, 0.6) + calc_weights_entropy(net.conv1, 0.6)

#         loss = criterion(outputs, targets) + calc_weights_entropy(net.conv6, 0.4, avg=True) + calc_weights_entropy(net.conv5, 0.4, avg=True)

#         loss = criterion(outputs, targets) + calc_weights_entropy(net.conv6, 0.4, avg=True) + calc_weights_entropy(net.conv5, 0.4, avg=True)

    
#         loss = criterion(outputs, targets) + probability_decay * (torch.norm(net.conv1.alpha, 2) + torch.norm(net.conv1.betta, 2)
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
    if writer is not None:
        writer.add_scalar("Loss/train", loss, epoch)
    print('{} seconds'.format(time.time() - t0))
    return (100.*correct/total), entropy_level


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

#             if net.sign_prob6.test_forward:
#                 outputs = net(inputs)
#             else:
#                 outputs, p = net(inputs)
      
#             if net.sign_prob6.test_forward and net.sign_prob5.test_forward:
#                 outputs = net(inputs)
#             elif net.sign_prob5.test_forward and not net.sign_prob6.test_forward:
#                 outputs, p6 = net(inputs)       
#             else:
#                 outputs, p6, p5 = net(inputs)

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
    
    
def layer_hist(name, x, writer, iteration=None, iter_list=None):             
    if iteration in iter_list:
        iteration_idx = str(iteration) if iteration is not None else ''
        writer.add_histogram(str(name) + " " + str(iteration_idx) + " distribution", x, 0, bins='auto')  

def id_generator(size=16, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def calc_avg_entropy (p, category, name, inner_iteration_train, writer):
    q = 1 - p + 1e-10            
    p = p + 1e-10            
    entropy = (-1) * ((p * torch.log(p)) + (q * torch.log(q)))
    entropy = torch.mean(entropy)            
    writer.add_scalar(str(category) + str(name), entropy, inner_iteration_train)
    entropy = (-1) * (p * torch.log(p))
    entropy = torch.mean(entropy)    
    writer.add_scalar(str(category) + "_only_p" + str(name) + "_only_p", entropy, inner_iteration_train)

def calc_avg_entropy3 (p1, p2, category, name, inner_iteration_train, writer):
    p3 = 1 - p1 - p2 + 1e-10                 
    p1 = p1 + 1e-10            
    p2 = p2 + 1e-10            
    entropy = (-1) * ((p1 * torch.log(p1)) + (p2 * torch.log(p2)) + (p3 * torch.log(p3)))
    entropy = torch.mean(entropy)            
    writer.add_scalar(str(category) + str(name), entropy, inner_iteration_train)
    entropy = (-1) * (p1 * torch.log(p1))
    entropy = torch.mean(entropy)    
    writer.add_scalar(str(category) + "_only_alpha" + str(name) + "_only_alpha", entropy, inner_iteration_train)
    entropy = (-1) * (p2 * torch.log(p2))
    entropy = torch.mean(entropy)    
    writer.add_scalar(str(category) + "_only_betta" + str(name) + "_only_betta", entropy, inner_iteration_train)

    
def get_p (x, output_sample):   
    if output_sample:
        z, p = x
    else:
        m, v, p = x 
    return p


def get_x (x, sign_layer):   
    test_forward = sign_layer.test_forward
    collect_stats = sign_layer.collect_stats
    output_sample = sign_layer.output_sample
    if test_forward or not collect_stats:
        x = x
    else:
        if output_sample:            
            z, p = x   
            x = z
        else:
            m, v, p = x
            x = m, v
    return x

 
def calc_m_v_sample (x, layer, N, name, writer, iteration, iter_list):
    if iteration in iter_list:
        test_samples = []
        with torch.no_grad():
            m, v = layer(x)
            m = m[0][0][0][0]
            v = v[0][0][0][0]
            for i in range(N):   
#                 print("compare_m_v", i)
                # rand weight
                layer.test_mode_switch(1,1)
                # calc output
                y = layer(x)
#                 test_samples.append(torch.unsqueeze(y[0][0][0][0],0))
                test_samples.append(y[0][0][0][0].data.cpu().numpy())                
#                 y = y[0][0][0][0]                
#                 writer.add_histogram(str(name) + " " + str(iteration) + " samples", y, 0, bins='auto')  
    
#             b = torch.Tensor(N).cuda()
#             torch.cat(test_samples, out=b)
#             print(b)
            test_samples = np.array(test_samples)
            writer.add_histogram(str(name) + " " + str(iteration) + " samples", test_samples, 0, bins='auto')              
#             writer.add_histogram(str(name) + " " + str(iteration) + " samples", x, torch.mean(b), bins='auto')  
#             writer.add_histogram(str(name) + " " + str(iteration) + " samples", x, 0, bins='auto')  

            m = test_samples.mean()
            v = test_samples.std()
            print(str(name) + "_" + str(iteration) + ": m samples is: ", m)
            print(str(name) + "_" + str(iteration) + ": v samples is: ", v)            
            layer.train_mode_switch()
            return m, v            
    else:
        return None, None        
            
def calc_m_v_analyt (x, layer, N, name, writer, iteration, iter_list):
    if iteration in iter_list:
        samples = []
        with torch.no_grad():
            m, v = layer(x)
            m = m[0][0][0][0]
            v = v[0][0][0][0]
            for i in range(N):
#                 print("compare_m_v_1", i)                
                # sample according to m/v
                epsilon = torch.normal(0, 1, size=m.size())
                r = m + epsilon * v
                r = torch.unsqueeze(r,0)
                samples.append(r.data.cpu().numpy())
#                 writer.add_histogram(str(name) + " " + str(iteration) + " analytics", r, 0, bins='auto')  

#             b = torch.Tensor(N).cuda()
#             torch.cat(samples, out=b)
            samples = np.array(samples)
            writer.add_histogram(str(name) + " " + str(iteration) + " analytics", samples, 0, bins='auto')  
#             writer.add_histogram(str(name) + " " + str(iteration) + " analytics", x, 0, bins='auto')  
            
            print(str(name) + "_" + str(iteration) + ": m analytics is: ", m)
            print(str(name) + "_" + str(iteration) + ": v analytics is: ", v)
            
            layer.train_mode_switch()
            return m, v
    else:
        return None, None

def compare_m_v (m_a, v_a, m_s, v_s, name, writer, iteration, iter_list):
    if iteration in iter_list:    
        m_ratio = m_a / m_s
        v_ratio = v_a / v_s
        writer.add_scalar(str(name) + "_m", m_ratio, iteration)        
        writer.add_scalar(str(name) + "_v", v_ratio, iteration)        


def calc_weights_entropy (layer, desired, avg=False, eps=1e-10):
    alpha_p = F.sigmoid(layer.alpha)           
    betta_p = F.sigmoid(layer.betta) * (1 - alpha_p) 
    gamma_p = 1 - alpha_p - betta_p
    alpha_p = alpha_p + eps
    betta_p = betta_p + eps
    gamma_p = gamma_p + eps
    entropy = (-1) * ((alpha_p * torch.log(alpha_p)) + (betta_p * torch.log(betta_p)) + (gamma_p * torch.log(gamma_p)))
    if avg:
        avg_entropy = torch.mean(entropy)    
        return torch.pow((avg_entropy - desired), 2)       
    else:
        reg_entropy = torch.sum(torch.pow((entropy - desired), 2)) 
    return reg_entropy

def calc_act_entropy (p, desired, avg=False, eps=1e-10):
    q = 1 - p + eps
    p = p + eps
    entropy = (-1) * ((p * torch.log(p)) + (q * torch.log(q)))
    if avg:
        avg_entropy = torch.mean(entropy)    
        return torch.pow((avg_entropy - desired), 2)       
    else:
        reg_entropy = torch.sum(torch.pow((entropy - desired), 2)) 
    return reg_entropy


def calc_average_entropy (p, eps=1e-10):
    q = 1 - p + eps
    p = p + eps
    entropy = (-1) * ((p * torch.log(p)) + (q * torch.log(q)))
    avg_entropy = torch.mean(entropy)    
    return avg_entropy       
  
        
        
        
        
        
        
        
