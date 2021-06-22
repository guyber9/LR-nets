'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
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

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

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

# def find_weights(w, my_prints=False):
#     if my_prints:
#         print("w: " + str(w))
#         print(w.size())
#         print(type(w))
#     # note: e^alpha + e^betta + e^gamma = 1
#     p_max = 0.95
#     p_min = 0.05
#     w_norm = w / torch.std(w)
#     e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
#     if my_prints:
#         print("e_alpha: " + str(e_alpha))
#     e_alpha = torch.clamp(e_alpha, p_min, p_max)
#     if my_prints:
#         print("e_alpha.clip: " + str(e_alpha))
#         print("e_alpha.size: " + str(e_alpha.size()))
#
#     # betta = 0.5 * (1 + (w_norm / (1 - alpha)))
#     e_betta = 0.5 * (w_norm - e_alpha + 1)
#     if my_prints:
#         print("e_betta: " + str(e_betta))
#     e_betta = torch.clamp(e_betta, p_min, p_max)
#     if my_prints:
#         print("e_betta.clip: " + str(e_betta))
#
#     alpha_prob = torch.log(e_alpha)
#     betta_prob = torch.log(e_betta)
#     gamma_prob = torch.log(torch.clamp((1 - e_alpha - e_betta), p_min, p_max))
#     if my_prints:
#         print("alpha_prob: " + str(alpha_prob))
#         print("betta_prob: " + str(betta_prob))
#         print("gamma_prob: " + str(gamma_prob))
#     alpha_prob = alpha_prob.detach().cpu().clone().numpy()
#     betta_prob = betta_prob.detach().cpu().clone().numpy()
#     gamma_prob = gamma_prob.detach().cpu().clone().numpy()
#     alpha_prob = np.expand_dims(alpha_prob, axis=-1)
#     betta_prob = np.expand_dims(betta_prob, axis=-1)
#     gamma_prob = np.expand_dims(gamma_prob, axis=-1)
#     theta = np.concatenate((alpha_prob, betta_prob, gamma_prob), axis=4)
#     if my_prints:
#         print("theta: " + str(theta))
#         print("theta.shape: " + str(np.shape(theta)))
#     return theta


def find_sigm_weights(w, my_prints=False):
    if my_prints:
        print("w: " + str(w))
        print(w.size())
        print(type(w))

    p_max = 0.95
    p_min = 0.05
    w_norm = w / torch.std(w)
    e_alpha = p_max - ((p_max - p_min) * torch.abs(w_norm))
    e_betta = 0.5 * (1 + (w_norm / (1 - e_alpha)))
    if my_prints:
        print("alpha: " + str(alpha))
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
