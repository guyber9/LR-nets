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
def train(net, criterion, epoch, device, trainloader, optimizer, args, f=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print("output: " + str(outputs))
        # print("targets: " + str(targets))
        loss = criterion(outputs, targets)
        # print("loss1: " + str(loss))
        if args.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        # print("loss1: " + str(loss))
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
        if f is not None:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()), file = f)
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
    print('{} seconds'.format(time.time() - t0))
    return (100.*correct/total)


def test(net, criterion, epoch, device, testloader, args, best_acc, best_epoch, test_mode=False, f=None):
    # global best_acc
    net.eval()
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

            print('\n' ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(testloader.dataset),
                100. * correct / len(testloader.dataset)))
            if f is not None:
                print('\n' + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(testloader.dataset),
                    100. * correct / len(testloader.dataset)), file=f)

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
        if not args.dont_save:
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
            torch.save(net.state_dict(), "saved_models/" + str(dataset_name) + str(net_type) + str(isBinary) + str(isVer2) + ".pt")
        best_acc = acc
        best_epoch = epoch
    if test_mode:
        best_acc = acc

    print("--> best accuracy is: " + str(best_acc) + " (epoch: " + str(best_epoch) + ")")
    if f is not None:
        print("--> best accuracy is: " + str(best_acc) + " (epoch: " + str(best_epoch) + ")", file=f)
    return best_acc, best_epoch, acc


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
#
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.
#
#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
#
#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')
#
#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time
#
#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)
#
#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')
#
#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))
#
#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()
#
# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)
#
#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f

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
                print (str(input_name) + "(" + str(i) + ", " + str(j) + ", " + str(m) + ": " + str(val3))