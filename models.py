from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import torch.nn as nn
import time
import layers as lrnet_nn
from utils import print_full_tensor, assertnan, collect_hist, collect_m_v, take_silce, output_hist, layer_hist, id_generator, calc_avg_entropy, calc_avg_entropy3, get_p, get_x, compare_m_v, calc_m_v_sample, calc_m_v_analyt
import numpy as np
from utils import test
import torch.autograd as autograd
import math

################
## MNIST nets ##
################

class FPNet(nn.Module):

    def __init__(self):
        super(FPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2) # 32 x 12 x 12
        x = F.relu(x)
        x = self.conv2(x) # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1) # 1024
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output


class LRNet(nn.Module):

    def __init__(self):
        super(LRNet, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 1)
        self.conv2 = lrnet_nn.LRnetConv2d(32, 64, 5, 1)
        # self.conv1 = MyBinaryConv2d(1, 32, 5, 1)
        # self.conv2 = MyBinaryConv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        self.bn1 = lrnet_nn.MyBatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = lrnet_nn.MyBatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2) # 32 x 12 x 12
        x = F.relu(x)
        x = self.conv2(x) # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1) # 1024
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)

    def inc_cntr(self):
        self.conv1.cntr = self.conv1.cntr + 1
        self.conv2.cntr = self.conv2.cntr + 1

    def rst_cntr(self):
        self.conv1.cntr = 0
        self.conv2.cntr = 0

    def update_collect_stats(self, new_val):
        self.bn1.update_collect_stats(new_val)
        self.bn2.update_collect_stats(new_val)

    def update_use_test_stats(self, new_val):
        self.bn1.update_use_test_stats(new_val)
        self.bn2.update_use_test_stats(new_val)

    def update_use_batch_stats(self, new_val):
        self.bn1.update_use_batch_stats(new_val)
        self.bn2.update_use_batch_stats(new_val)

        
class FPNet_ver2(nn.Module):

    def __init__(self):
        super(FPNet_ver2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
#         x = self.bn1(x)
        x = self.conv2(x)  # 64 x 16 x 16 / 64 x 20 x 20
#         x = self.bn2(x)    
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 64 x 8 x 8 / # 64 x 10 x 10
        x = torch.flatten(x, 1)  # 1024
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x      
        return output

    def train_mode_switch(self):
        return

    def test_mode_switch(self, options, tickets):
        return
    
    def use_batch_stats_switch(self, new_val):
        return

    def collect_stats_switch(self, new_val):
        return
        
class LRNet_ver2(nn.Module):

    def __init__(self, writer):
        super(LRNet_ver2, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 2 ,output_sample=False)
        # self.conv2 = lrnet_nn.LRnetConv2d_ver2(32, 32, 5, 1, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d_ver2(32, 64, 5, 2, output_sample=False) # True worked
        self.sign_prob = lrnet_nn.LRnet_sign_prob() # True worked

        # self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 1)
        # self.conv2 = lrnet_nn.LRnetConv2d(32, 32, 5, 1)
        # self.conv3 = lrnet_nn.LRnetConv2d(32, 64, 5, 1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)

#         self.fc1 = nn.Linear(1024, 512)
        self.fc1 = lrnet_nn.LRnetLinear(1024, 512)        
        # self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 10)

        self.bn1 = lrnet_nn.LRBatchNorm2d(32)
        # self.bn2 = lrnet_nn.LRBatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
#         if self.writer is not None:
        self.writer = writer
        self.iteration = 0
        self.test_iteration = 0
        
    def forward(self, x):
        assertnan(x, "x0")
        x = self.conv1(x)  # 32 x 24 x 24
#         if not self.conv1.test_forward:
#             if self.writer is not None: 
#                 m, v = x
#                 self.writer.add_scalar("m1 mean", torch.mean(m), self.iteration)
#                 self.writer.add_scalar("m1 std", torch.std(m), self.iteration)
#                 self.writer.add_scalar("m1 max", torch.max(m), self.iteration)
#                 self.writer.add_scalar("m1 min", torch.min(m), self.iteration)
#                 self.writer.add_scalar("v1 mean", torch.mean(v), self.iteration)
#                 self.writer.add_scalar("v1 std", torch.mean(v), self.iteration)
#                 self.writer.add_scalar("v1 max", torch.max(m), self.iteration)
#                 self.writer.add_scalar("v1 min", torch.min(m), self.iteration)                
#                 self.writer.add_scalar("max(m1-v1)", torch.max(m/v), self.iteration)
#                 self.writer.add_scalar("min(m1-v1)", torch.min(m/v), self.iteration)   
#                 self.writer.add_scalar("mean(m1-v1)", torch.mean(m/v), self.iteration)
#                 self.writer.add_scalar("std(m1-v1)", torch.std(m/v), self.iteration)
#                 if (self.iteration%400 == 0):                 
#                     self.writer.add_histogram('alpha distribution', self.conv1.alpha + self.iteration, self.iteration)
#                     self.writer.add_histogram('betta distribution', self.conv1.betta + self.iteration, self.iteration)
#                 self.writer.add_scalar("alpha mean", torch.mean(self.conv1.alpha), self.iteration)
#                 self.writer.add_scalar("alpha std", torch.std(self.conv1.alpha), self.iteration)
#                 self.writer.add_scalar("alpha max", torch.max(self.conv1.alpha), self.iteration)
#                 self.writer.add_scalar("alpha min", torch.min(self.conv1.alpha), self.iteration)  
#                 if (self.iteration == 0):
#                     self.writer.add_histogram('alpha distribution 0', self.conv1.alpha, 0)
#                 if (self.iteration == 100):
#                     self.writer.add_histogram('alpha distribution 100', self.conv1.alpha, 0)
#                 if (self.iteration == 1000):
#                     self.writer.add_histogram('alpha distribution 1000', self.conv1.alpha, 0)
#                 if (self.iteration == 1500):
#                     self.writer.add_histogram('alpha distribution 1500', self.conv1.alpha, 0)
#                 if (self.iteration == 2000):
#                     self.writer.add_histogram('alpha distribution 2000', self.conv1.alpha, 0)
#                 if (self.iteration == 2500):
#                     self.writer.add_histogram('alpha distribution 2500', self.conv1.alpha, 0)
#                 if (self.iteration == 3000):
#                     self.writer.add_histogram('alpha distribution 3000', self.conv1.alpha, 0)
#                 if (self.iteration == 4000):
#                     self.writer.add_histogram('alpha distribution 4000', self.conv1.alpha, 0)     
#         else:
#             if (self.test_iteration%100 == 0):                             
#                 self.writer.add_histogram('x1 test distribution', x + self.test_iteration, self.test_iteration)    
#             self.test_iteration = self.test_iteration + 1
                    
        # m,v = x
        # print("m1: ", m)
        # print("v1: ", v)
        # assertnan(m, "m1")
        # assertnan(v, "v1")
#         x = self.bn1(x)
        # m,v = x
        # assertnan(m, "mbn1")
        # assertnan(v, "vbn1")
        # x = self.conv2(x)  # 32 x 20 x 20
        # m,v = x
        # print("m2: ", m)
        # print("v2: ", v)
        # m,v = x
        # assertnan(m, "m2")
        # assertnan(v, "v2")
        # x = self.bn2(x)
        # m,v = x
        # assertnan(m, "mbn2")
        # assertnan(v, "vbn2")
        x = self.conv2(x)  # 64 x 16 x 16 / 64 x 20 x 20
#         if not self.conv1.test_forward:
#             if self.writer is not None: 
#                 self.writer.add_scalar("x2 mean", torch.mean(x), self.iteration)
#                 self.writer.add_scalar("x2 std", torch.std(x), self.iteration)
#                 self.writer.add_scalar("x2 max", torch.max(x), self.iteration)
#                 self.writer.add_scalar("x2 min", torch.min(x), self.iteration)  
#                 self.writer.add_histogram('x2 distribution', x + self.iteration, self.iteration)  
#                 if (self.iteration == 0):
#                     self.writer.add_histogram('x2 distribution 0', x, 0)
#                 if (self.iteration == 100):
#                     self.writer.add_histogram('x2 distribution 100', x, 0)
#                 if (self.iteration == 1000):
#                     self.writer.add_histogram('x2 distribution 1000', x, 0)
#                 if (self.iteration == 1500):
#                     self.writer.add_histogram('x2 distribution 1500', x, 0)
#                 if (self.iteration == 2000):
#                     self.writer.add_histogram('x2 distribution 2000', x, 0)
#                 if (self.iteration == 2500):
#                     self.writer.add_histogram('x2 distribution 2500', x, 0)
#                 if (self.iteration == 3000):
#                     self.writer.add_histogram('x2 distribution 3000', x, 0)
#                 if (self.iteration == 4000):
#                     self.writer.add_histogram('x2 distribution 4000', x, 0)                  
#         assertnan(x, "x3")
#         x = self.bn3(x)
#         assertnan(x, "bn3")
#         x = F.relu(x) # worked
        x = self.sign_prob(x)   
#         assertnan(x, "x3")    
#         x = F.max_pool2d(x, 2)  # 64 x 8 x 8 / # 64 x 10 x 10
#         x = torch.flatten(x, 1)  # 1024
        if self.fc1.test_forward:
            x = torch.flatten(x, 1)
        else: 
            m, v = x
            m = torch.flatten(m, 1)
            v = torch.flatten(v, 1)
            x = m, v
#         x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)        
        x = self.fc2(x)
        output = x
        assertnan(output, "output")
        if not self.conv1.test_forward:
            if self.writer is not None: 
                self.iteration = self.iteration+1           
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        # self.conv2.train_mode_switch()
        self.conv2.train_mode_switch()
        self.sign_prob.train_mode_switch()       
        self.bn1.train_mode_switch()
        # self.bn2.train_mode_switch()
        self.fc1.train_mode_switch()        

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        # self.conv2.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.sign_prob.test_mode_switch(options, tickets)               
        self.bn1.test_mode_switch()
        # self.bn2.test_mode_switch()
        self.fc1.test_mode_switch(options, tickets)        

    def use_batch_stats_switch(self, new_val):
        self.bn1.use_batch_stats_switch(new_val)

    def collect_stats_switch(self, new_val):
        self.bn1.collect_stats_switch(new_val)

        
class LRNet_ver2XXX(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_ver2XXX, self).__init__()
        
        self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, stride=2, padding=0, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d(32, 64, 5, stride=2, padding=0, output_sample=False)

        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_sample=True, collect_stats=True)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_sample=True, collect_stats=True)

        self.collect_all_stats = True        
        self.bn_s = False
        self.gumbel = True
        self.gain = True
        
        self.args = args        
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
            self.inner_iteration_test = 0
            self.inner_iteration_train = 0
        self.tensorboard_train = False
        self.tensorboard_test = False                 

        self.fc1 = nn.Linear(1024, 512)
#         self.fc1 = lrnet_nn.LRnetLinear(1024, 512)        
        self.fc2 = nn.Linear(512, 10)

        self.bn3 = lrnet_nn.LRBatchNorm2d(32)
    
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid_func = torch.nn.Sigmoid()   
        self.iterations_list = [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
        
    def forward(self, x):
        cin = x
        x = self.conv1(x)
        
        x = self.sign_prob1(x)        
        if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
            print("LRNet_CIFAR10_ver2_not_sample iteration_train:", self.iteration_train)
            print("LRNet_CIFAR10_ver2_not_sample inner_iteration_train:", self.inner_iteration_train)
            p = get_p(x, self.sign_prob1.output_sample)
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign1_entropy", self.iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv1.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv1.betta) * (1 - alpha_p) + 1e-10  
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_1", self.inner_iteration_train, self.writer)
            if self.collect_all_stats:
                layer_hist("sign1_p/sign1", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv1/iteraion", self.writer, self.iteration_train, self.iterations_list)    
        x = get_x (x, self.sign_prob1)
        
        cin = x
        x = self.conv2(x)
        x = self.sign_prob2(x)

        if self.tensorboard_train and self.sign_prob2.collect_stats and not self.sign_prob2.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob2.output_sample)             
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign2_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv2.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv2.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_2", self.inner_iteration_train, self.writer)             
            if self.collect_all_stats:
                layer_hist("sign2_p/sign2", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv2/iteraion", self.writer, self.iteration_train, self.iterations_list)    
            self.tensorboard_train = False
            self.inner_iteration_test = self.inner_iteration_test + 1
            self.inner_iteration_train = self.inner_iteration_train + 1                
                
        x = get_x (x, self.sign_prob2)  

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)        
        x = self.fc2(x)
        output = x        
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
   

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(1, 1)
        self.conv2.test_mode_switch(1, 1)
        self.sign_prob1.test_mode_switch(1, 1)
        self.sign_prob2.test_mode_switch(1, 1)

        
##################
## CIFAR10 nets ##
##################

class FPNet_CIFAR10(nn.Module):

    def __init__(self):
        super(FPNet_CIFAR10, self).__init__()
        
        self.only_1_fc = False

        self.conv1 = nn.Conv2d(3, 128, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        
        if self.only_1_fc:
            self.dropout1 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(8192, 10)
        else:
            self.dropout1 = nn.Dropout(0.5)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(8192, 1024)
            self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 3
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x) # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 128 x 16 x 16
        x = F.relu(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x) # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2) # 256 x 8 x 8
        x = F.relu(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x) # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2) # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1) # 8192

        if self.only_1_fc:
#             x = self.dropout1(x)
            x = self.fc1(x)  # 8192 -> 1024            
        else:
            x = self.dropout1(x)
            x = self.fc1(x)  # 8192 -> 1024
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x) # 1024 -> 10
            output = x
        return output

class VGG_SMALL(nn.Module):

    def __init__(self):
        super(VGG_SMALL, self).__init__()
        self.wide=1.0
        self.conv1 = nn.Conv2d(3, int(128*self.wide), 3, 1, padding=1)
        self.conv2 = nn.Conv2d(int(128*self.wide), int(128*self.wide), 3, 1, padding=1)
        self.conv3 = nn.Conv2d(int(128*self.wide), int(256*self.wide), 3, 1, padding=1)
        self.conv4 = nn.Conv2d(int(256*self.wide), int(256*self.wide), 3, 1, padding=1)
        self.conv5 = nn.Conv2d(int(256*self.wide), int(256*self.wide), 3, 1, padding=1)
        self.conv6 = nn.Conv2d(int(256*self.wide), int(256*self.wide), 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(128*self.wide))
        self.bn2 = nn.BatchNorm2d(int(128*self.wide))
        self.bn3 = nn.BatchNorm2d(int(256*self.wide))
        self.bn4 = nn.BatchNorm2d(int(256*self.wide))
        self.bn5 = nn.BatchNorm2d(int(256*self.wide))
        self.bn6 = nn.BatchNorm2d(int(256*self.wide))
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(4096*self.wide), 10) # 8192 # 4096 # 2048

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 3
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x) # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 128 x 16 x 16
        x = F.relu(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x) # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2) # 256 x 8 x 8
        x = F.relu(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x) # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2) # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x) 
        output = x
        return output    
    
class LRNet_CIFAR10(nn.Module):

    def __init__(self, writer=None):
        super(LRNet_CIFAR10, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, 1, padding=1)
        self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, 1, padding=1)
        self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, 1, padding=1)
        self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, 1, padding=1)
        self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, 1, padding=1)
        self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
#         self.bn1 = lrnet_nn.MyBatchNorm2d(num_features=128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn2 = lrnet_nn.MyBatchNorm2d(num_features=128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn3 = lrnet_nn.MyBatchNorm2d(num_features=256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn4 = lrnet_nn.MyBatchNorm2d(num_features=256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn5 = lrnet_nn.MyBatchNorm2d(num_features=512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
#         self.bn6 = lrnet_nn.MyBatchNorm2d(num_features=512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout2 = nn.Dropout(0.5)

        self.dropout4 = nn.Dropout(0.3) # 0.2 was 93.13
        self.dropout6 = nn.Dropout(0.3) # 0.2 was 93.13
        self.dropout1 = nn.Dropout(0.3)

        self.dropout3 = nn.Dropout(0.25) # 0.2 was 93.13
        self.dropout5 = nn.Dropout(0.25) # 0.2 was 93.13
        self.dropout7 = nn.Dropout(0.25) # 0.2 was 93.13
        
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
        self.tensorboard_train = False
        self.tensorboard_test = False         
        
    def forward(self, x): 
        net_input = x
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        
        first_time_1 = True
        if self.tensorboard_test and (self.writer is not None): 
            N = 10000#0
            output_hist("x1", x, net_input, self.conv1, self.writer, N)             
            
        x = self.bn1(x)  # <- problematic batchnoram (?)
        x = F.relu(x)
        x = self.dropout3(x) # ???

        net_input = x                   
        x = self.conv2(x)  # 128 x 32 x 32

        first_time_1 = True
        if self.tensorboard_test and (self.writer is not None): 
            output_hist("x2", x, net_input, self.conv2, self.writer, N)             
            
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout5(x)  # ???
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout7(x)  # ???

        net_input = x                   

        x = self.conv6(x)  # 512 x 8 x 8

        first_time_1 = True
        if self.tensorboard_test and (self.writer is not None): 
            output_hist("x6", x, net_input, self.conv6, self.writer, N)                         
            self.tensorboard_test = False

        x = self.bn6(x)
        x = F.max_pool2d(x, 2)  # 512 x 4 x 4 (= 8192)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)  # 8192
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output

    # def forward(self, x):
    #     x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
    #     # print("x1: " + str(x))
    #     # print("x1 isnan: " + str(torch.isnan(x).any()))
    #     x = self.bn1(x)  # <- problematic batchnoram (?)
    #     # print("bn1 isnan: " + str(torch.isnan(x).any()))
    #     x = F.relu(x)
    #     x = self.dropout3(x)
    #     # print("xrelu1 isnan: " + str(torch.isnan(x).any()))
    #     # print("start here")
    #     x = self.conv2(x)  # 128 x 32 x 32
    #     # print_full_tensor(x, "x2 full")
    #     # print("x2 isnan: " + str(torch.isnan(x).any()))
    #     # print("x2: " + str(x))
    #     x = self.bn2(x)
    #     # print("bn2 isnan: " + str(torch.isnan(x).any()))
    #     # print("bn2: " + str(x))
    #     x = F.max_pool2d(x, 2)  # 128 x 16 x 16
    #     x = F.relu(x)
    #     x = self.dropout4(x)
    #
    #     x = self.conv3(x)  # 256 x 16 x 16
    #     x = self.bn3(x)
    #     x = F.relu(x)
    #     x = self.dropout5(x)
    #     x = self.conv4(x)  # 256 x 16 x 16
    #     x = self.bn4(x)
    #     x = F.max_pool2d(x, 2)  # 256 x 8 x 8
    #     x = F.relu(x)
    #     x = self.dropout6(x)
    #
    #     x = self.conv5(x)  # 512 x 8 x 8
    #     x = self.bn5(x)
    #     x = F.relu(x)
    #     x = self.dropout7(x)
    #     x = self.conv6(x)  # 512 x 8 x 8
    #     x = self.bn6(x)
    #     x = F.max_pool2d(x, 2)  # 512 x 4 x 4 (= 8192)
    #     x = F.relu(x)
    #
    #     x = torch.flatten(x, 1)  # 8192
    #     x = self.dropout1(x)
    #     x = self.fc1(x)  # 8192 -> 1024
    #     x = F.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)  # 1024 -> 10
    #     output = x
    #     return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)

    def inc_cntr(self):
        self.conv1.cntr = self.conv1.cntr + 1
        self.conv2.cntr = self.conv2.cntr + 1
        self.conv3.cntr = self.conv3.cntr + 1
        self.conv4.cntr = self.conv4.cntr + 1
        self.conv5.cntr = self.conv5.cntr + 1
        self.conv6.cntr = self.conv6.cntr + 1

    def rst_cntr(self):
        self.conv1.cntr = 0
        self.conv2.cntr = 0
        self.conv3.cntr = 0
        self.conv4.cntr = 0
        self.conv5.cntr = 0
        self.conv6.cntr = 0

    def update_collect_stats(self, new_val):
        self.bn1.update_collect_stats(new_val)
        self.bn2.update_collect_stats(new_val)
        self.bn3.update_collect_stats(new_val)
        self.bn4.update_collect_stats(new_val)
        self.bn5.update_collect_stats(new_val)
        self.bn6.update_collect_stats(new_val)

    def update_use_test_stats(self, new_val):
        self.bn1.update_use_test_stats(new_val)
        self.bn2.update_use_test_stats(new_val)
        self.bn3.update_use_test_stats(new_val)
        self.bn4.update_use_test_stats(new_val)
        self.bn5.update_use_test_stats(new_val)
        self.bn6.update_use_test_stats(new_val)

    def update_use_batch_stats(self, new_val):
        self.bn1.update_use_batch_stats(new_val)
        self.bn2.update_use_batch_stats(new_val)
        self.bn3.update_use_batch_stats(new_val)
        self.bn4.update_use_batch_stats(new_val)
        self.bn5.update_use_batch_stats(new_val)
        self.bn6.update_use_batch_stats(new_val)


class LRNet_CIFAR10_ver2(nn.Module):

    def __init__(self, writer=None):
        super(LRNet_CIFAR10_ver2, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d_ver2(128, 128, 3, stride=2, padding=1, output_sample=False)
        self.conv3 = lrnet_nn.LRnetConv2d_ver2(128, 256, 3, stride=1, padding=1, output_sample=True)
        self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=2, padding=1, output_sample=False)
#         self.conv4 = lrnet_nn.LRnetConv2d_ver2(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv5 = lrnet_nn.LRnetConv2d_ver2(256, 512, 3, stride=1, padding=1, output_sample=False)
        self.conv6 = lrnet_nn.LRnetConv2d_ver2(512, 512, 3, stride=2, padding=1, output_sample=True)
        self.sign_prob = lrnet_nn.LRnet_sign_prob()
        self.bn1 = lrnet_nn.LRBatchNorm2d(128, affine=False)
        self.bn2 = lrnet_nn.LRBatchNorm2d(128, affine=False)
#         self.bn3 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn4 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn5 = lrnet_nn.LRBatchNorm2d(512, affine=False)
#         self.bn6 = lrnet_nn.LRBatchNorm2d(512, affine=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        # self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(8192, 1024)
#         self.fc1 = lrnet_nn.LRnetLinear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

#         self.dropout1 = nn.Dropout(0.5)

        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        num_of_epochs = 10
        test_iter_per_batch = 10 
        train_iter_per_batch = 196
        
        self.train_last_epoch = num_of_epochs*train_iter_per_batch - 1
        self.test_last_epoch = num_of_epochs*test_iter_per_batch - 1
        
    def forward(self, x):
        net_input = x
        x = self.conv1(x)

#         take_silce('layer1_full_output', x)    
                
#         if self.tensorboard_train and (self.writer is not None):     
#             m,v=x  

#             with torch.no_grad():            
#                 self.conv1.test_mode_switch()
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv1.device)
#                 train_x = m + epsilon * v
#                 test_x = self.conv1(net_input)
#                 self.conv1.train_mode_switch()
#             self.writer.add_scalar("conv1 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)     
                            
#             take_silce('layer1_full_output', x)    
                
#             collect_m_v(self.writer, 1, x, self.iteration_train)
#             self.writer.add_scalar("m1 [0,0,0] mean", torch.mean(m[:,0,0,0]), self.iteration_train)     
#             self.writer.add_scalar("v1 [0,0,0] mean", torch.mean(v[:,0,0,0]), self.iteration_train)     
#             if (self.iteration_train == 0):
#                 self.writer.add_histogram("m1 [0,0,0] iteration" + str(0) + " distribution", m[:,0,0,0], 0)              
#                 self.writer.add_histogram("v1 [0,0,0] iteration" + str(0) + " distribution", v[:,0,0,0], 0) 
#             if (self.iteration_train == self.train_last_epoch):
#                 self.writer.add_histogram("m1 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", m[:,0,0,0], 0)              
#                 self.writer.add_histogram("v1 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", v[:,0,0,0], 0)                                  
        N = 10000
#         first_time_1 = True
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 5, 50, 99, 150, 199]:        
                t0 = time.time()
                print ("iteration_test 1: ", self.iteration_test)
                output_hist("x_1", x, net_input, self.conv1, self.writer, N, self.iteration_test)         
                print('{} seconds'.format(time.time() - t0))

#             for i in range(N):
#                 if first_time_1:
#                     y = x[0,0,0,0].view(1)
#                     first_time_1 = False                
#                 else:
#                     self.conv1.test_mode_switch(1, 1)
#                     z = self.conv1(net_input)
#                     y = torch.cat((y,z[0,0,0,0].view(1)))
#             self.writer.add_histogram("x_1 [0,0,0] distribution", y, 0, bins='auto')                  

            
#             self.writer.add_scalar("x1 mean", torch.mean(x), self.iteration_test)    
#             self.writer.add_scalar("x1 std", torch.std(x), self.iteration_test) 
#             collect_hist(0, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
# #             collect_hist(50, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
# #             collect_hist(100, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
#             collect_hist(self.test_last_epoch, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)         
            
#             self.writer.add_scalar("x1 [0,0,0] mean", torch.mean(x[:,0,0,0]), self.iteration_test)     
#             self.writer.add_scalar("x1 [0,0,0] std", torch.std(x[:,0,0,0]), self.iteration_test)                   
#             if (self.iteration_test == 0):
#                 self.writer.add_histogram("x1 [0,0,0] iteration" + str(0) + " distribution", x[:,0,0,0], 0)  
#             if (self.iteration_test == self.test_last_epoch):
#                 self.writer.add_histogram("x1 [0,0,0] iteration" + str(self.test_last_epoch) + " distribution", x[:,0,0,0], 0)  

# #             if (self.iteration%3 == 0):                 
# #                 self.writer.add_histogram('alpha1 distribution', self.conv1.alpha + self.iteration, self.iteration)
# #         x = self.bn1(x)
        

#         if self.tensorboard_train and (self.writer is not None):      
#             with torch.no_grad():            
#                 m, v = x
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv2.device)
#                 net_input = m + epsilon * v
        net_input = x
        x = self.conv2(x)  # 128 x 32 x 32

#         first_time_1 = True
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 99, 199]:        
                t0 = time.time()
                print ("iteration_test 2: ", self.iteration_test)                
                output_hist("x2", x, net_input, self.conv2, self.writer, N, self.iteration_test)       
                print('{} seconds'.format(time.time() - t0))

#             for i in range(N):
#                 if first_time_1:
#                     y = x[0,0,0,0].view(1)
#                     first_time_1 = False                
#                 else:
#                     self.conv2.test_mode_switch(1, 1)
#                     z = self.conv2(net_input)
#                     y = torch.cat((y,z[0,0,0,0].view(1)))
#             self.writer.add_histogram("x_2 [0,0,0] distribution", y, 0, bins='auto')  
            
#         if self.tensorboard_train and (self.writer is not None):                            
#             take_silce('layer2_full_output', x)    
# #         x = self.bn2(x)
#         # x = F.relu(x)

#         if self.tensorboard_train and (self.writer is not None):      
#             m, v = x
#             with torch.no_grad():            
#                 self.conv2.test_mode_switch()
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv2.device)
#                 train_x = m + epsilon * v
#                 test_x = self.conv2(net_input)
#                 self.conv2.train_mode_switch()
#             self.writer.add_scalar("conv2 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)        

#         if self.tensorboard_train and (self.writer is not None):      
#             with torch.no_grad():            
#                 m, v = x
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv3.device)
#                 net_input = m + epsilon * v

        net_input = x                
        x = self.conv3(x)  # 256 x 16 x 16
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 99, 199]:        
                output_hist("x3", x, net_input, self.conv3, self.writer, N, self.iteration_test)  
#         if self.tensorboard_train and (self.writer is not None):      
#             m, v = x
#             with torch.no_grad():            
#                 self.conv3.test_mode_switch()
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv3.device)
#                 train_x = m + epsilon * v
#                 test_x = self.conv3(net_input)
#                 self.conv3.train_mode_switch()
#             self.writer.add_scalar("conv3 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)                       
        
#         if self.tensorboard_train and (self.writer is not None):                            
#             take_silce('layer3_full_output', x)        
# #         x = self.bn3(x)
#         # x = F.relu(x)
        
        x = self.bn3(x)  # 256 x 16 x 16
        x = F.relu(x)  # 256 x 16 x 16

        net_input = x
        x = self.conv4(x)  # 256 x 16 x 16
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 99, 199]:        
                output_hist("x4", x, net_input, self.conv4, self.writer, N, self.iteration_test)  
            
#         if self.tensorboard_train and (self.writer is not None):                            
#             take_silce('layer4_full_output', x)         
# #         x = self.bn4(x)

        net_input = x
        x = self.conv5(x)  # 512 x 8 x 8
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 99, 199]:        
                output_hist("x5", x, net_input, self.conv5, self.writer, N, self.iteration_test)          
#         if self.tensorboard_train and (self.writer is not None):     
#             take_silce('layer5_full_output', x)
# #         x = self.bn5(x)
#         # x = F.relu(x)
        
#         if self.tensorboard_train and (self.writer is not None):      
#             with torch.no_grad():            
#                 m, v = x
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv6.device)
#                 net_input = m + epsilon * v
        net_input = x
        x = self.conv6(x)  # 512 x 8 x 8
        if self.tensorboard_test and (self.writer is not None): 
            if self.iteration_test in [0, 5, 50, 99, 150, 199]:        
                t0 = time.time()
                print ("iteration_test 6: ", self.iteration_test)                                
                output_hist("x6", x, net_input, self.conv6, self.writer, N, self.iteration_test) 
                print('{} seconds'.format(time.time() - t0))
                self.tensorboard_test = False

#         if self.tensorboard_test and (self.writer is not None): 
#             for i in range(N):
#                 if first_time_1:
#                     full_input = x 
#                     y = x[0,0,0,0].view(1)
#                     first_time_1 = False                
#                 else:
#                     self.conv6.test_mode_switch(1, 1)
#                     z = self.conv6(net_input)
#                     y = torch.cat((y,z[0,0,0,0].view(1)))
#             self.writer.add_histogram("x_6 [0,0,0] distribution", y, 0)           
#             self.tensorboard_test = False
    
#         if self.tensorboard_train and (self.writer is not None):      
#             m, v = x
#             with torch.no_grad():            
#                 self.conv6.test_mode_switch()
#                 epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv6.device)
#                 train_x = m + epsilon * v
#                 test_x = self.conv6(net_input)
#                 self.conv6.train_mode_switch()
#             self.writer.add_scalar("conv6 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)                
            
#             take_silce('layer6_full_output', x)   
# #             exit(1)
# #         if self.tensorboard_train and (self.writer is not None):     
# #             m,v=x
# #             collect_m_v(self.writer, 6, x, self.iteration_train)
# #             self.writer.add_scalar("m6 [0,0,0] mean", torch.mean(m[:,0,0,0]), self.iteration_train)     
# #             self.writer.add_scalar("v6 [0,0,0] mean", torch.mean(v[:,0,0,0]), self.iteration_train)     
# #             if (self.iteration_train == 0):
# #                 self.writer.add_histogram("m6 [0,0,0] iteration" + str(0) + " distribution", m[:,0,0,0], 0)              
# #                 self.writer.add_histogram("v6 [0,0,0] iteration" + str(0) + " distribution", v[:,0,0,0], 0)    
# #             if (self.iteration_train == self.train_last_epoch):
# #                 self.writer.add_histogram("m6 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", m[:,0,0,0], 0)              
# #                 self.writer.add_histogram("v6 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", v[:,0,0,0], 0)                 
# #         if self.tensorboard_test and (self.writer is not None):     
# #             collect_hist(0, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
# #             collect_hist(50, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
# #             collect_hist(100, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
# #             collect_hist(self.test_last_epoch, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
# # 
# #             self.writer.add_scalar("x6 [0,0,0] mean", torch.mean(x[:,0,0,0]), self.iteration_test)
# #             self.writer.add_scalar("x6 [0,0,0] std", torch.std(x[:,0,0,0]), self.iteration_test)  
# #             if (self.iteration_test == 0):
# #                 self.writer.add_histogram("x6 [0,0,0] iteration" + str(0) + " distribution", x[:,0,0,0], 0)  
# #             if (self.iteration_test == self.test_last_epoch):
# #                 self.writer.add_histogram("x6 [0,0,0] iteration" + str(self.test_last_epoch) + " distribution", x[:,0,0,0], 0)              
        x = self.bn6(x)
#         x = self.sign_prob(x)        
        x = F.relu(x)
                   
#         if self.fc1.test_forward:
#             x = torch.flatten(x, 1)  # 8192
#         else: 
#             m, v = x
#             m = torch.flatten(m, 1)  # 8192
#             v = torch.flatten(v, 1)  # 8192
#             x = m, v
        x = torch.flatten(x, 1)  # 8192

        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        
#         if self.tensorboard_test and (self.writer is not None):
#             self.iteration_test = self.iteration_test + 1        
#         if self.tensorboard_train and (self.writer is not None):
#             self.iteration_train = self.iteration_train + 1        

        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        self.sign_prob.train_mode_switch()
        self.bn1.train_mode_switch()
        self.bn2.train_mode_switch()
#         self.bn3.train_mode_switch()
        self.bn4.train_mode_switch()
        self.bn5.train_mode_switch()
#         self.bn6.train_mode_switch()
#         self.fc1.train_mode_switch()        

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.sign_prob.test_mode_switch(options, tickets)        
        self.bn1.test_mode_switch()
        self.bn2.test_mode_switch()
#         self.bn3.test_mode_switch()
        self.bn4.test_mode_switch()
        self.bn5.test_mode_switch()
#         self.bn6.test_mode_switch()
#         self.fc1.test_mode_switch(options, tickets)        

    def use_batch_stats_switch(self, new_val):
        self.bn1.use_batch_stats_switch(new_val)
        self.bn2.use_batch_stats_switch(new_val)
#         self.bn3.use_batch_stats_switch(new_val)
        self.bn4.use_batch_stats_switch(new_val)
        self.bn5.use_batch_stats_switch(new_val)
#         self.bn6.use_batch_stats_switch(new_val)

    def collect_stats_switch(self, new_val):
        self.bn1.collect_stats_switch(new_val)
        self.bn2.collect_stats_switch(new_val)
#         self.bn3.collect_stats_switch(new_val)
        self.bn4.collect_stats_switch(new_val)
        self.bn5.collect_stats_switch(new_val)
#         self.bn6.collect_stats_switch(new_val)
        
class FPNet_CIFAR10_ver2(nn.Module):

    def __init__(self):
        super(FPNet_CIFAR10_ver2, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
        self.bn1 = lrnet_nn.MyBatchNorm2d(128)
        self.bn2 = lrnet_nn.MyBatchNorm2d(128)
        self.bn3 = lrnet_nn.MyBatchNorm2d(256)
        self.bn4 = lrnet_nn.MyBatchNorm2d(256)
        self.bn5 = lrnet_nn.MyBatchNorm2d(512)
        self.bn6 = lrnet_nn.MyBatchNorm2d(512)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)  # 512 x 8 x 8      
        x = F.relu(x)
        x = torch.flatten(x, 1)  # 8192
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output

    def train_mode_switch(self):
        return

    def test_mode_switch(self, options, tickets):
        return

    def use_batch_stats_switch(self, new_val):
        return

    def collect_stats_switch(self, new_val):
        return
    

class LRNet_CIFAR10_ver2_debug(nn.Module):

    def __init__(self, writer=None):
        super(LRNet_CIFAR10_ver2_debug, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d_ver2(128, 128, 3, stride=2, padding=1, output_sample=False)
        self.conv3 = lrnet_nn.LRnetConv2d_ver2(128, 256, 3, stride=1, padding=1, output_sample=False)
#         self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv4 = lrnet_nn.LRnetConv2d_ver2(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv5 = lrnet_nn.LRnetConv2d_ver2(256, 512, 3, stride=1, padding=1, output_sample=False)
        self.conv6 = lrnet_nn.LRnetConv2d_ver2(512, 512, 3, stride=2, padding=1, output_sample=True)
#         self.sign_prob = lrnet_nn.LRnet_sign_prob()
#         self.bn6 = lrnet_nn.LRBatchNorm2d(512, affine=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(8192, 1024)
#         self.fc1 = lrnet_nn.LRnetLinear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self.bn1 = lrnet_nn.LRBatchNorm2d(128, affine=False)
        self.bn2 = lrnet_nn.LRBatchNorm2d(128, affine=False)
        self.bn3 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn4 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn5 = lrnet_nn.LRBatchNorm2d(512, affine=False)    
    
    def forward(self, x):
        net_input = x
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.conv5(x) 
        x = self.conv6(x)
#         x = self.sign_prob(x)  
        x = self.bn6(x)
        x = F.relu(x)
#         if self.fc1.test_forward:
#             x = torch.flatten(x, 1)  # 8192
#         else: 
#             m, v = x
#             m = torch.flatten(m, 1)  # 8192
#             v = torch.flatten(v, 1)  # 8192
#             x = m, v
        x = torch.flatten(x, 1)  # 8192
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output

# here1    
class LRNet_CIFAR10_ver2X(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_CIFAR10_ver2X, self).__init__()
        
        self.conv1_is_normal = False

        self.args = args
        
        self.sampled_last_layer = True # freeze_last_layer = False  |  dont_freeze_last_layer = True
        self.bn_s = False
        self.gumbel = True        
        self.gumble_last_layer = False # True if not self.sampled_last_layer else False
        self.bn_last_layer = False or self.bn_s # True if not self.sampled_last_layer else False
        self.gain = True 
        
        self.only_1_fc = args.only_1_fc
        self.dropout = False
        self.zero_act = 0
        self.tau = 1.0
        
        self.pool = False
        self.stride = 1 if self.pool else 2
        if self.pool:
            self.pool2 = lrnet_nn.LRnetAvgPool2d()
            self.pool4 = lrnet_nn.LRnetAvgPool2d()
            if self.sampled_last_layer:
                self.pool6 = nn.AvgPool2d(2, stride=2)                
            else:
                self.pool6 = lrnet_nn.LRnetAvgPool2d()
                
        if self.only_1_fc:
            self.fc1_output_chan = 10
        else:
            self.fc1_output_chan = 1024 

        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)

        if self.gumbel:
            self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=True if self.sampled_last_layer else False)          
        else:
            self.conv2 = lrnet_nn.LRnetConv2d_ver2X(128, 128, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d_ver2X(128, 256, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d_ver2X(256, 256, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d_ver2X(256, 512, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d_ver2X(512, 512, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=True if self.sampled_last_layer else False)

#         self.conv2 = lrnet_nn.LRnetConv2d_ver2X(128, 128, 3, stride=2, padding=1, output_gain=self.gain, output_sample=False)
#         self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
#         self.conv4 = lrnet_nn.LRnetConv2d_ver2X(256, 256, 3, stride=2, padding=1, output_gain=self.gain, output_sample=False)
#         self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
#         self.conv6 = lrnet_nn.LRnetConv2d_ver2X(512, 512, 3, stride=2, padding=1, output_gain=self.gain, output_sample=True if self.sampled_last_layer else False)
            
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=False, bn_layer = self.bn_s)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_chan=512, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=False, bn_layer = self.bn_s)
#         self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_chan=512, output_sample=self.gumbel, collect_stats=False if self.sampled_last_layer else True, bn_layer = self.bn_s)
        self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_chan=512, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumble_last_layer, collect_stats=False, bn_layer = self.bn_last_layer)
 
        self.bn6 = nn.BatchNorm2d(512)

        if self.sampled_last_layer or self.gumble_last_layer:
            self.fc1 = nn.Linear(8192, self.fc1_output_chan)
        else:
            self.fc1 = lrnet_nn.LRnetLinear(8192, self.fc1_output_chan)
        if not self.only_1_fc:
            self.fc2 = nn.Linear(1024, 10)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
    
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
            self.inner_iteration_test = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        num_of_epochs = 10
        test_iter_per_batch = 10 
        train_iter_per_batch = 196
        
        self.train_last_epoch = num_of_epochs*train_iter_per_batch - 1
        self.test_last_epoch = num_of_epochs*test_iter_per_batch - 1
        
        self.net_id = id_generator(16)
                
    def forward(self, x):
        x = self.conv1(x)
#         if self.tensorboard_train and (self.writer is not None): 
#             if self.iteration_train in [0, 1, 2]:        
#                 layer_hist("x1", x, self.writer, self.iteration_test)   
        x = self.sign_prob1(x)
        if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
            print("start collect stats for sign1")            
            if self.sign_prob1.output_sample:
                z, p = x   
            else:
                m, v, p = x                   
            p = p + 1e-10            
            entropy = (-1) * p * torch.log2(p)
            entropy = torch.mean(entropy)
            self.writer.add_scalar("train/sign1_entropy", entropy, self.inner_iteration_test)
            if self.iteration_train in [0, 10, 50, 100, 130, 150, 200, 250, 299, 350, 399, 449]:    
                layer_hist("sign1", p, self.writer, self.iteration_test) 
#             self.tensorboard_train = False
            print("end collect stats for sign1")            

        if self.sign_prob1.test_forward or not self.sign_prob1.collect_stats:
            x = x
        else:
            if self.sign_prob1.output_sample:            
                z, p = x   
                x = z
            else:
                m, v, p = x
                x = m, v

        x = self.conv2(x)
        if self.pool:
            x = self.pool2(x)
        x = self.sign_prob2(x)
        x = self.conv3(x)
        x = self.sign_prob3(x)
        x = self.conv4(x)
        if self.pool:
            x = self.pool4(x)        
        x = self.sign_prob4(x)
        x = self.conv5(x)
        x = self.sign_prob5(x)
        
        if self.tensorboard_train and self.sign_prob5.collect_stats and not self.sign_prob5.test_forward and (self.writer is not None): 
            print("start collect stats for sign5")
            if self.sign_prob5.output_sample:
                z, p = x   
            else:
                m, v, p = x                
            p = p + 1e-10
            entropy = (-1) * p * torch.log2(p)
            entropy = torch.mean(entropy)
            self.writer.add_scalar("train/sign5_entropy", entropy, self.inner_iteration_test)
            if self.iteration_train in [0, 10, 50, 100, 130, 150, 200, 250, 299, 350, 399, 449]:    
                layer_hist("sign5", p, self.writer, self.iteration_test) 
            if self.sampled_last_layer:
                self.tensorboard_train = False
                self.inner_iteration_test = self.inner_iteration_test + 1
            print("end collect stats for sign5")
            
        if self.sign_prob5.test_forward or not self.sign_prob5.collect_stats:
            x = x
        else:
            if self.sign_prob5.output_sample:            
                z, p = x   
                x = z
            else:
                m, v, p = x
                x = m, v
            
        x = self.conv6(x)
        if self.pool:
            x = self.pool6(x) 
            
        if self.sampled_last_layer:
            x = self.bn6(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)  # 8192
        else:                  
            x = self.sign_prob6(x)    
            if self.tensorboard_train and self.sign_prob6.collect_stats and not self.sign_prob6.test_forward and (self.writer is not None): 
                print("start collect stats for sign6")                
                if self.sign_prob6.output_sample:
                    z, p = x   
                else:
                    m, v, p = x                    
                p = p + 1e-10                
                entropy = (-1) * p * torch.log2(p)
                entropy = torch.mean(entropy)
                self.writer.add_scalar("train/sign6_entropy", entropy, self.inner_iteration_test)
                if self.iteration_train in [0, 10, 50, 100, 130, 150, 200, 250, 299, 350, 399, 449]:    
                    layer_hist("sign6", p, self.writer, self.iteration_test) 
                self.tensorboard_train = False
                self.inner_iteration_test = self.inner_iteration_test + 1
                print("end collect stats for sign6")
                
            if self.sign_prob6.test_forward:
                x = torch.flatten(x, 1)  # 8192            
            elif self.gumble_last_layer:
                if self.sign_prob6.collect_stats:
                    z, p = x
                    x = torch.flatten(z, 1)  # 8192
                else:
                    x = torch.flatten(x, 1)  # 8192
            else: 
                if self.sign_prob6.collect_stats:
                    if self.sign_prob6.output_sample:      
                        z, p = x   
                        x = z
                    else:
                        m, v, p = x
                        x = m, v                    
                else:
                    m, v = x
                    
                if not self.gumble_last_layer:
                    m,v = x
                    
                m = torch.flatten(m, 1)  # 8192
                v = torch.flatten(v, 1)  # 8192
                x = m, v
        if self.only_1_fc:
            if self.dropout:
                x = self.dropout2(x)
            x = self.fc1(x)  # 8192 -> 10        
        else:
    #         x = self.dropout1(x)
            x = self.fc1(x)  # 8192 -> 1024
            x = F.relu(x)
    #         x = self.dropout2(x)
            x = self.fc2(x)  # 1024 -> 10
        output = x
        
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
        self.sign_prob3.train_mode_switch()
        self.sign_prob4.train_mode_switch()
        self.sign_prob5.train_mode_switch()
        self.sign_prob6.train_mode_switch()
        if not self.sampled_last_layer and not self.gumble_last_layer:
            self.fc1.train_mode_switch()    
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.sampled_last_layer:
                self.pool6.train_mode_switch()           

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.sign_prob1.test_mode_switch(options, tickets)    
        self.sign_prob2.test_mode_switch(options, tickets)
        self.sign_prob3.test_mode_switch(options, tickets)
        self.sign_prob4.test_mode_switch(options, tickets)
        self.sign_prob5.test_mode_switch(options, tickets)
        self.sign_prob6.test_mode_switch(options, tickets)
        if not self.sampled_last_layer and not self.gumble_last_layer:        
            self.fc1.test_mode_switch(options, tickets)      
        if self.pool:
            self.pool2.test_mode_switch(options, tickets)
            self.pool4.test_mode_switch(options, tickets)
            if not self.sampled_last_layer:
                self.pool6.test_mode_switch(options, tickets)             

    def freeze_layer(self, conv_layer, sign_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None, update_tau=None):
        print('==> Freezing...') 
        
#         self.tau = 0.8 * self.tau
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau
        
        sign_layer.test_mode_switch(1, 1)
                
        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 

#         for param in sign_layer.bn.parameters():
#             param.requires_grad = False 
            
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)

        net.load_state_dict(torch.load(model_id))
            

    def freeze_last_layer(self, conv_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing Last Layer...') 
        
#         self.tau = 0.1
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau

        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        for param in conv_layer.parameters():
            param.requires_grad = False 
        
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)            
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)          
                
        net.load_state_dict(torch.load(model_id))
            
    def train_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.train_mode_switch()
            self.sign_prob1.train_mode_switch()    
        if freeze_mask[1] is not 0:
            self.conv2.train_mode_switch()
            self.sign_prob2.train_mode_switch()
        if freeze_mask[2] is not 0:
            self.conv3.train_mode_switch()
            self.sign_prob3.train_mode_switch()
        if freeze_mask[3] is not 0:
            self.conv4.train_mode_switch()
            self.sign_prob4.train_mode_switch()
        if freeze_mask[4] is not 0:
            self.conv5.train_mode_switch()
            self.sign_prob5.train_mode_switch()
        if freeze_mask[5] is not 0:
            self.conv6.train_mode_switch()        
            self.sign_prob6.train_mode_switch()   
        if not self.sampled_last_layer and not self.gumble_last_layer:
            self.fc1.train_mode_switch()        
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.sampled_last_layer:
                self.pool6.train_mode_switch()      
                
    def test_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.test_mode_switch(1, 1)
            self.sign_prob1.test_mode_switch(1, 1)    
        if freeze_mask[1] is not 0:
            self.conv2.test_mode_switch(1, 1)
            self.sign_prob2.test_mode_switch(1, 1)
        if freeze_mask[2] is not 0:
            self.conv3.test_mode_switch(1, 1)
            self.sign_prob3.test_mode_switch(1, 1)
        if freeze_mask[3] is not 0:
            self.conv4.test_mode_switch(1, 1)
            self.sign_prob4.test_mode_switch(1, 1)
        if freeze_mask[4] is not 0:
            self.conv5.test_mode_switch(1, 1)
            self.sign_prob5.test_mode_switch(1, 1)
        if freeze_mask[5] is not 0:
            self.conv6.test_mode_switch(1, 1)        
            self.sign_prob6.test_mode_switch(1, 1)   
        if not self.sampled_last_layer and not self.gumble_last_layer:
            self.fc1.test_mode_switch(1, 1)     
        if self.pool:
            self.pool2.test_mode_switch(1, 1)
            self.pool4.test_mode_switch(1, 1)
            if not self.sampled_last_layer:
                self.pool6.test_mode_switch(1, 1)   
            
class LRNet_CIFAR10_ver2XXX(nn.Module):

    def __init__(self, writer=None):
        super(LRNet_CIFAR10_ver2XXX, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, stride=2, padding=1, output_sample=False)
        self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, stride=1, padding=1, output_sample=False)
        self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, stride=1, padding=1, output_sample=False)
        self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, stride=2, padding=1, output_sample=True)
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_sample=True)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_sample=True)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_sample=True)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_sample=True)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_sample=True)
        self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_sample=True)

        self.bn6 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(8192, 1024)
#         self.fc1 = lrnet_nn.LRnetLinear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        num_of_epochs = 10
        test_iter_per_batch = 10 
        train_iter_per_batch = 196
        
        self.train_last_epoch = num_of_epochs*train_iter_per_batch - 1
        self.test_last_epoch = num_of_epochs*test_iter_per_batch - 1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sign_prob1(x)
        x = self.conv2(x)
        x = self.sign_prob2(x)
        x = self.conv3(x)
        x = self.sign_prob3(x)
        x = self.conv4(x)
        x = self.sign_prob4(x)
        x = self.conv5(x)
        x = self.sign_prob5(x)
        x = self.conv6(x)
        x = self.bn6(x)
#         x = self.sign_prob(x)        
        x = F.relu(x)
                   
#         if self.fc1.test_forward:
#             x = torch.flatten(x, 1)  # 8192
#         else: 
#             m, v = x
#             m = torch.flatten(m, 1)  # 8192
#             v = torch.flatten(v, 1)  # 8192
#             x = m, v
        x = torch.flatten(x, 1)  # 8192

        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
        self.sign_prob3.train_mode_switch()
        self.sign_prob4.train_mode_switch()
        self.sign_prob5.train_mode_switch()
        self.sign_prob6.train_mode_switch()
#         self.fc1.train_mode_switch()        

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.sign_prob1.test_mode_switch(options, tickets)    
        self.sign_prob2.test_mode_switch(options, tickets)
        self.sign_prob3.test_mode_switch(options, tickets)
        self.sign_prob4.test_mode_switch(options, tickets)
        self.sign_prob5.test_mode_switch(options, tickets)
        self.sign_prob6.test_mode_switch(options, tickets)        
#         self.fc1.test_mode_switch(options, tickets)        

    def freeze_layer(self, conv_layer, sign_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None):
        print('==> Freezing...') 
        sign_layer.test_mode_switch(1, 1)
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 
       
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                print('==> saving model:', 'saved_models/freezed_model.pt')
                torch.save(net.state_dict(), 'saved_models/freezed_model.pt')

        net.load_state_dict(torch.load('saved_models/freezed_model.pt'))
            

    def freeze_last_layer(self, conv_layer, trials, net, criterion, device, trainloader, args, f=None):
        print('==> Freezing Last Layer...') 
        for param in conv_layer.parameters():
            param.requires_grad = False 
        
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)            
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                print('==> saving model:', 'saved_models/freezed_model.pt')
                torch.save(net.state_dict(), 'saved_models/freezed_model.pt')          
                
        net.load_state_dict(torch.load('saved_models/freezed_model.pt'))
            
    def train_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.train_mode_switch()
            self.sign_prob1.train_mode_switch()    
        if freeze_mask[1] is not 0:
            self.conv2.train_mode_switch()
            self.sign_prob2.train_mode_switch()
        if freeze_mask[2] is not 0:
            self.conv3.train_mode_switch()
            self.sign_prob3.train_mode_switch()
        if freeze_mask[3] is not 0:
            self.conv4.train_mode_switch()
            self.sign_prob4.train_mode_switch()
        if freeze_mask[4] is not 0:
            self.conv5.train_mode_switch()
            self.sign_prob5.train_mode_switch()
        if freeze_mask[5] is not 0:
            self.conv6.train_mode_switch()        
            self.sign_prob6.train_mode_switch()   
            
#         self.fc1.train_mode_switch()        

    def test_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.test_mode_switch(1, 1)
            self.sign_prob1.test_mode_switch(1, 1)    
        if freeze_mask[1] is not 0:
            self.conv2.test_mode_switch(1, 1)
            self.sign_prob2.test_mode_switch(1, 1)
        if freeze_mask[2] is not 0:
            self.conv3.test_mode_switch(1, 1)
            self.sign_prob3.test_mode_switch(1, 1)
        if freeze_mask[3] is not 0:
            self.conv4.test_mode_switch(1, 1)
            self.sign_prob4.test_mode_switch(1, 1)
        if freeze_mask[4] is not 0:
            self.conv5.test_mode_switch(1, 1)
            self.sign_prob5.test_mode_switch(1, 1)
        if freeze_mask[5] is not 0:
            self.conv6.test_mode_switch(1, 1)        
            self.sign_prob6.test_mode_switch(1, 1)   
        
#         self.fc1.test_mode_switch(options, tickets)     


# here2  
class LRNet_CIFAR10_ver2_sample(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_CIFAR10_ver2_sample, self).__init__()

        self.args = args
        self.sampled_last_layer = True # always True here2
        
        self.bn_s = False
        self.gumbel = True        
        self.gain = True 
        
        self.dropout = False
        self.zero_act = 0
        self.tau = 1.0
        
        self.pool = False
        
        self.stride = 2
#         self.stride = 1 if self.pool else 2
#         if self.pool:
#             self.pool2 = lrnet_nn.LRnetAvgPool2d()
#             self.pool4 = lrnet_nn.LRnetAvgPool2d()
#             if self.sampled_last_layer:
#                 self.pool6 = nn.AvgPool2d(2, stride=2)                
#             else:
#                 self.pool6 = lrnet_nn.LRnetAvgPool2d()
                
        self.fc1_output_chan = 10

        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)

        if self.gumbel:
            self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=True)          
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=False, bn_layer = self.bn_s)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, bn_layer = self.bn_s)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_chan=512, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=False, bn_layer = self.bn_s)
 
        self.bn6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(8192, self.fc1_output_chan)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
    
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
            self.inner_iteration_test = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        self.net_id = id_generator(16)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.sign_prob1(x)
        
        if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
            print("start collect stats for sign1")            
            if self.sign_prob1.output_sample:
                z, p = x   
            else:
                m, v, p = x                   
            p = p + 1e-10            
            entropy = (-1) * p * torch.log2(p)
            entropy = torch.mean(entropy)
            self.writer.add_scalar("train/sign1_entropy", entropy, self.inner_iteration_test)
            if self.iteration_train in [0, 10, 50, 100, 130, 150, 200, 250, 299, 350, 399, 449]:    
                layer_hist("sign1", p, self.writer, self.iteration_test) 
#             self.tensorboard_train = False
            print("end collect stats for sign1")            

        if self.sign_prob1.test_forward or not self.sign_prob1.collect_stats:
            x = x
        else:
            if self.sign_prob1.output_sample:            
                z, p = x   
                x = z
            else:
                m, v, p = x
                x = m, v

        x = self.conv2(x)
        if self.pool:
            x = self.pool2(x)
        x = self.sign_prob2(x)
        x = self.conv3(x)
        x = self.sign_prob3(x)
        x = self.conv4(x)
        if self.pool:
            x = self.pool4(x)        
        x = self.sign_prob4(x)
        x = self.conv5(x)
        x = self.sign_prob5(x)
        
        if self.tensorboard_train and self.sign_prob5.collect_stats and not self.sign_prob5.test_forward and (self.writer is not None): 
            print("start collect stats for sign5")
            if self.sign_prob5.output_sample:
                z, p = x   
            else:
                m, v, p = x                
            p = p + 1e-10
            entropy = (-1) * p * torch.log2(p)
            entropy = torch.mean(entropy)
            self.writer.add_scalar("train/sign5_entropy", entropy, self.inner_iteration_test)
            if self.iteration_train in [0, 10, 50, 100, 130, 150, 200, 250, 299, 350, 399, 449]:    
                layer_hist("sign5", p, self.writer, self.iteration_test) 
                self.tensorboard_train = False
                self.inner_iteration_test = self.inner_iteration_test + 1
            print("end collect stats for sign5")
            
        if self.sign_prob5.test_forward or not self.sign_prob5.collect_stats:
            x = x
        else:
            if self.sign_prob5.output_sample:            
                z, p = x   
                x = z
            else:
                m, v, p = x
                x = m, v
            
        x = self.conv6(x)
        if self.pool:
            x = self.pool6(x) 
            
        x = self.bn6(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)  # 8192

        if self.dropout:
            x = self.dropout2(x)
        x = self.fc1(x)  # 8192 -> 10        

        output = x
        
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
        self.sign_prob3.train_mode_switch()
        self.sign_prob4.train_mode_switch()
        self.sign_prob5.train_mode_switch()
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.sign_prob1.test_mode_switch(options, tickets)    
        self.sign_prob2.test_mode_switch(options, tickets)
        self.sign_prob3.test_mode_switch(options, tickets)
        self.sign_prob4.test_mode_switch(options, tickets)
        self.sign_prob5.test_mode_switch(options, tickets)
        if self.pool:
            self.pool2.test_mode_switch(options, tickets)
            self.pool4.test_mode_switch(options, tickets)

    def freeze_layer(self, conv_layer, sign_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing...') 
        
#         self.tau = 0.8 * self.tau
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau
        
        sign_layer.test_mode_switch(1, 1)
                
        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 

#         for param in sign_layer.bn.parameters():
#             param.requires_grad = False 
            
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)

        net.load_state_dict(torch.load(model_id))
            

    def freeze_last_layer(self, conv_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing Last Layer...') 
        
#         self.tau = 0.1
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau

        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        for param in conv_layer.parameters():
            param.requires_grad = False 
        
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)            
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)          
                
        net.load_state_dict(torch.load(model_id))
            
    def train_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.train_mode_switch()
            self.sign_prob1.train_mode_switch()    
        if freeze_mask[1] is not 0:
            self.conv2.train_mode_switch()
            self.sign_prob2.train_mode_switch()
        if freeze_mask[2] is not 0:
            self.conv3.train_mode_switch()
            self.sign_prob3.train_mode_switch()
        if freeze_mask[3] is not 0:
            self.conv4.train_mode_switch()
            self.sign_prob4.train_mode_switch()
        if freeze_mask[4] is not 0:
            self.conv5.train_mode_switch()
            self.sign_prob5.train_mode_switch()
        if freeze_mask[5] is not 0:
            self.conv6.train_mode_switch()        
#             self.sign_prob6.train_mode_switch()   
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
                
    def test_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            self.conv1.test_mode_switch(1, 1)
            self.sign_prob1.test_mode_switch(1, 1)    
        if freeze_mask[1] is not 0:
            self.conv2.test_mode_switch(1, 1)
            self.sign_prob2.test_mode_switch(1, 1)
        if freeze_mask[2] is not 0:
            self.conv3.test_mode_switch(1, 1)
            self.sign_prob3.test_mode_switch(1, 1)
        if freeze_mask[3] is not 0:
            self.conv4.test_mode_switch(1, 1)
            self.sign_prob4.test_mode_switch(1, 1)
        if freeze_mask[4] is not 0:
            self.conv5.test_mode_switch(1, 1)
            self.sign_prob5.test_mode_switch(1, 1)
        if freeze_mask[5] is not 0:
            self.conv6.test_mode_switch(1, 1)        
#             self.sign_prob6.test_mode_switch(1, 1)   
        if self.pool:
            self.pool2.test_mode_switch(1, 1)
            self.pool4.test_mode_switch(1, 1)

            
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None                       
            
class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      
        
        self.sparsity = 0.5
        self.my_requires_grad = False

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = self.my_requires_grad
        
    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)        
        w = self.weight * subnet
        print("w:", w)
        print("nonzero:", torch.count_nonzero(w))        
        return F.linear(x, w, self.bias)

# here3
class LRNet_CIFAR10_ver2_not_sample(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_CIFAR10_ver2_not_sample, self).__init__()
        
        print("LRNet_CIFAR10_ver2_not_sample")

        self.args = args
        self.sampled_last_layer = False # always False here3
        
        self.collect_all_stats = False
        
        self.gumbel_ = True

        self.wide = 1.00
        
        self.bn_s = False
        self.gumbel = True
        self.gumble_last_layer = True # True if not self.sampled_last_layer else False
        self.bn_last_layer = True or self.bn_s # True if not self.sampled_last_layer else False
        self.gain = True
        
        self.lrgain = False

        self.supermask = False         
        
        self.pool = False
        self.conv1_is_normal = False

        self.dropout = False
        
        self.zero_act = 0
        self.tau = 0.01
                
        self.l6 = 512
        
        self.stride = 1 if self.pool else 2
        if self.pool:
            self.pool2 = lrnet_nn.LRnetAvgPool2d()
            self.pool4 = lrnet_nn.LRnetAvgPool2d()
#             if self.gumble_last_layer:
#                 self.pool6 = nn.AvgPool2d(2, stride=2)                
#             else:
            self.pool6 = lrnet_nn.LRnetAvgPool2d()
                
        if self.wide > 1.0:
            self.flat_chan = int(8192 * self.wide)
        else:
            self.flat_chan = 8192 # 32768 # 16384 # 8192 # 4096 # 2048         
                
        if args.cifar100:
            self.fc1_output_chan = 100
        else:
            self.fc1_output_chan = 10

        if self.conv1_is_normal:
            self.conv1 = nn.Conv2d(3, 128, 3, 1, padding=1)  
            self.bn1 = nn.BatchNorm2d(128)
        else:
            self.conv1 = lrnet_nn.LRnetConv2d(3, int(128*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)

        if self.gumbel:
            self.conv2 = lrnet_nn.LRnetConv2d(int(128*self.wide), int(128*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d(int(128*self.wide), int(256*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d(int(256*self.wide), int(256*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d(int(256*self.wide), int(self.l6*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d(int(self.l6*self.wide), int(self.l6*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)          
            
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_chan=int(128*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_chan=int(128*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel_, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_chan=int(256*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_chan=int(256*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel_, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_chan=int(self.l6*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        
        self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_chan=int(self.l6*self.wide), tau=self.tau, zero_act=self.zero_act, output_sample=self.gumble_last_layer, collect_stats=True, bn_layer = self.bn_last_layer)
 
        if self.lrgain:
            self.gain1 = lrnet_nn.LRgain(num_features=int(128*self.wide))
            self.gain2 = lrnet_nn.LRgain(num_features=int(128*self.wide))
            self.gain3 = lrnet_nn.LRgain(num_features=int(256*self.wide))
            self.gain4 = lrnet_nn.LRgain(num_features=int(256*self.wide))
            self.gain5 = lrnet_nn.LRgain(num_features=int(512*self.wide))
            self.gain6 = lrnet_nn.LRgain(num_features=int(512*self.wide))

        if self.gumble_last_layer:
            if self.supermask:
                self.fc1 = SupermaskLinear(self.flat_chan, self.fc1_output_chan)                
            else:
                self.fc1 = nn.Linear(self.flat_chan, self.fc1_output_chan)
        else:
            self.fc1 = lrnet_nn.LRnetLinear(self.flat_chan, self.fc1_output_chan)

#         self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
    
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
            self.inner_iteration_test = 0
            self.inner_iteration_train = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        self.net_id = id_generator(16)
        self.sigmoid_func = torch.nn.Sigmoid()   
        self.iterations_list = [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
#         self.iterations_list = [0, 10, 50, 100, 200, 250, 350, args.epochs-1]
#         self.iterations_list1 = [0, 10, 50, 100, 200, 250, 270, 290, 310, 330, 350, args.epochs-1]
        self.iterations_list1 = self.iterations_list
        
    def forward(self, x):
#         qwery = x
#         self.tensorboard_train = False     
# #         self.test_mode_switch(1,1)        
#         for i in range(5):
#         x = qwery
#         print("------------------------------------------------------")
        cin = x
        x = self.conv1(x)
        
        if self.conv1_is_normal:   
            x = self.bn1(x)
            x = F.relu(x)            
        else:
            if self.lrgain:
                x = self.gain1(x)        
            x = self.sign_prob1(x)        
            if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
#                 print("self.sign_prob1 STATS")                
#                 x = self.sign_prob1(x, self.writer, self.iteration_train, self.iterations_list, 'bn1')                        
                print("LRNet_CIFAR10_ver2_not_sample iteration_train:", self.iteration_train)
                print("LRNet_CIFAR10_ver2_not_sample inner_iteration_train:", self.inner_iteration_train)
                p = get_p(x, self.sign_prob1.output_sample)
                p = p + 1e-10
                calc_avg_entropy (p, "sign_entropy", "/sign1_entropy", self.iteration_train, self.writer)
                alpha_p = self.sigmoid_func(self.conv1.alpha) + 1e-10            
                betta_p = self.sigmoid_func(self.conv1.betta) * (1 - alpha_p) + 1e-10  
                calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_1", self.inner_iteration_train, self.writer)
                if self.collect_all_stats:
                    layer_hist("sign1_p/sign1", p, self.writer, self.iteration_train, self.iterations_list) 
                    m_s, v_s = calc_m_v_sample (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                    m_a, v_a = calc_m_v_analyt (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                    compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv1/iteraion", self.writer, self.iteration_train, self.iterations_list)    
#             else:
#                 x = self.sign_prob1(x)                        
                    

            x = get_x (x, self.sign_prob1)
        
        cin = x
        x = self.conv2(x)
        if self.pool:
            x = self.pool2(x)

        if self.lrgain:            
            x = self.gain2(x)                    
        x = self.sign_prob2(x)

        if self.tensorboard_train and self.sign_prob2.collect_stats and not self.sign_prob2.test_forward and (self.writer is not None): 
#             x = self.sign_prob1(x, self.writer, self.iteration_train, self.iterations_list, 'bn2')                                
            p = get_p(x, self.sign_prob2.output_sample)             
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign2_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv2.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv2.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_2", self.inner_iteration_train, self.writer)             
            if self.collect_all_stats:
                layer_hist("sign2_p/sign2", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv2/iteraion", self.writer, self.iteration_train, self.iterations_list)   
#             else:
#                 x = self.sign_prob2(x)                

        x = get_x (x, self.sign_prob2)   
        cin = x
        x = self.conv3(x)
        if self.lrgain:        
            x = self.gain3(x)                
        x = self.sign_prob3(x)        

        if self.tensorboard_train and self.sign_prob3.collect_stats and not self.sign_prob3.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob3.output_sample)                            
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign3_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv3.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv3.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_3", self.inner_iteration_train, self.writer) 
            if self.collect_all_stats:
                layer_hist("sign3_p/sign3", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv3, 2000, 'conv3_clt/conv3', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv3, 2000, 'conv3_clt/conv3', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv3/iteraion", self.writer, self.iteration_train, self.iterations_list)    

        x = get_x (x, self.sign_prob3)   
        cin = x        
        x = self.conv4(x)
        if self.pool:
            x = self.pool4(x) 
        if self.lrgain:            
            x = self.gain4(x)                    
        x = self.sign_prob4(x)        

        if self.tensorboard_train and self.sign_prob4.collect_stats and not self.sign_prob4.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob4.output_sample)    
            p = p + 1e-10   
            print("calc_avg_entropy start")
            calc_avg_entropy (p, "sign_entropy", "/sign4_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv4.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv4.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_4", self.inner_iteration_train, self.writer) 
            if self.collect_all_stats:
                layer_hist("sign4_p/sign4", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv4, 2000, 'conv4_clt/conv4', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv4, 2000, 'conv4_clt/conv4', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv4/iteraion", self.writer, self.iteration_train, self.iterations_list)    

#             self.tensorboard_train = False
#             self.inner_iteration_test = self.inner_iteration_test + 1
#             self.inner_iteration_train = self.inner_iteration_train + 1

        x = get_x (x, self.sign_prob4)    

#         x = torch.flatten(x, 1)  # 16384                                   
#         output = self.fc1(x)  # 16384 -> 10 
#         return output

        cin = x
        x = self.conv5(x)
        if self.lrgain:        
            x = self.gain5(x)                            
        x = self.sign_prob5(x)

        if self.sign_prob5.test_forward:
            p5 = x    
        else:
            p5 = get_p(x, self.sign_prob5.output_sample)    


        if self.tensorboard_train and self.sign_prob5.collect_stats and not self.sign_prob5.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob5.output_sample)    
            p = p + 1e-10  
            calc_avg_entropy (p, "sign_entropy", "/sign5_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv5.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv5.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_5", self.inner_iteration_train, self.writer)          
            if self.collect_all_stats:            
                layer_hist("sign5_p/sign5", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv5, 2000, 'conv5_clt/conv5', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv5, 2000, 'conv5_clt/conv5', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv5/iteraion", self.writer, self.iteration_train, self.iterations_list)    

#             self.tensorboard_train = False
#             self.inner_iteration_test = self.inner_iteration_test + 1
#             self.inner_iteration_train = self.inner_iteration_train + 1

        x = get_x (x, self.sign_prob5)    

#         x = torch.flatten(x, 1)  # 32768                    
#         output = self.fc1(x)  # 32768 -> 10 
#         return output

#         if i == 0:
#             prev_tnsor_5 = torch.flatten(x, 1)
#         f = prev_tnsor_5 - torch.flatten(x, 1)            
#         prev_tnsor_5 = torch.flatten(x, 1)            
#         print("f size:", f.size())
#         for idx, val in enumerate(f[0]):
#             self.writer.add_scalar("conv5/feature_map_conv5_" + str(i), val, idx*100)        

        cin = x
        x = self.conv6(x)
        if self.pool:
            x = self.pool6(x) 
        if self.lrgain:            
            x = self.gain6(x)                                
        x = self.sign_prob6(x) 

        if self.tensorboard_train and self.sign_prob6.collect_stats and not self.sign_prob6.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob6.output_sample)    
            p = p + 1e-10  
            calc_avg_entropy (p, "sign_entropy", "/sign6_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv6.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv6.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_6", self.inner_iteration_train, self.writer)             
            layer_hist("sign6_p/sign6", p, self.writer, self.iteration_train, self.iterations_list)        
            if self.collect_all_stats:            
#                 layer_hist("sign6_p/sign6", p, self.writer, self.iteration_train, self.iterations_list)        
                m_s, v_s = calc_m_v_sample (cin, self.conv6, 2000, 'conv6_clt/conv6', self.writer, self.iteration_train, self.iterations_list1)
                m_a, v_a = calc_m_v_analyt (cin, self.conv6, 2000, 'conv6_clt/conv6', self.writer, self.iteration_train, self.iterations_list1)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv6/iteraion", self.writer, self.iteration_train, self.iterations_list1)

            self.tensorboard_train = False
            self.inner_iteration_test = self.inner_iteration_test + 1
            self.inner_iteration_train = self.inner_iteration_train + 1
#             exit(1)

        if self.sign_prob6.test_forward:
            x = torch.flatten(x, 1)  # 8192            
        elif self.gumble_last_layer:
            if self.sign_prob6.collect_stats:
                z, p = x
                x = torch.flatten(z, 1)  # 8192
            else:
                x = torch.flatten(x, 1)  # 8192
        else: 
            if self.sign_prob6.collect_stats:
                if self.sign_prob6.output_sample:      
                    z, p = x   
                    x = z
                    self.p = p
                else:
                    m, v, p = x
                    x = m, v                    
            else:
                m, v = x

            if not self.gumble_last_layer:
                m,v = x

            m = torch.flatten(m, 1)  # 8192
            v = torch.flatten(v, 1)  # 8192
            x = m, v

        if self.dropout:
            x = self.dropout2(x)

#         if i == 0:
#             prev_tnsor_6 = x
#         f = prev_tnsor_6 - x            
#         prev_tnsor_6 = torch.flatten(x, 1)               
#         for idx, val in enumerate(f[0]):
# #                 print(idx, val)
#             self.writer.add_scalar("conv6/eature_map_conv6_" + str(i), val, idx*100)        
# #             exit(1)


        x = self.fc1(x)  # 8192 -> 10 
    
#         # ###########################################################                
#         if self.tensorboard_train and self.sign_prob6.collect_stats and not self.sign_prob6.test_forward and (self.writer is not None):   
#             if self.inner_iteration_train in [0, 1, 5, 10, 200, 300, 350, 400, 450]:
#                 #             W = self.fc1.weight[0, None]
#                 my_p = torch.flatten(p, 1)
#     #             print ("p[0] size:", my_p[0].size())
#     #             print ("W size:", W[0].size())
#         #         exit(1)

#                 ip = my_p[0] 
#                 iq = 1 - ip + 1e-10            
#                 ip = ip + 1e-10            
#                 entropy = (-1) * ((ip * torch.log(ip)) + (iq * torch.log(iq)))    
#                 entropy = entropy.data.cpu().numpy()
#     #             W = W.data.cpu().numpy()
#     #             with open('entropy_calc/' + str('W_0') + '.npy', 'wb') as f:
#     #                 np.save(f, W)     
#                 with open('entropy_calc/' + str('entropy_fc1_') + str(self.iteration_train) + '.npy', 'wb') as f:
#                     np.save(f, entropy)          

#                 for i in range(0,10):
#                     W = self.fc1.weight[i, None]
#                     W = W.data.cpu().numpy()
#                     name = 'W_' + str(i) + '_' + str(self.iteration_train)
#                     with open('entropy_calc/' + name + '.npy', 'wb') as f:
#                         np.save(f, W)  
#             self.tensorboard_train = False
#             self.inner_iteration_test = self.inner_iteration_test + 1
#             self.inner_iteration_train = self.inner_iteration_train + 1                    
#     #         exit(1)
#         # ###########################################################    

#             print("feature_map:", x[0])
#             for idx, val in enumerate(x[0]):
# #                 print(idx, val)
#                 self.writer.add_scalar("outputs_layer" + str(i), val, idx)        

        output = x

# #             print("output size:", output.size())
# #             print("output size:", output[0])
#         print(" F.softmax(output):", F.softmax(output[0]))
#         _, predicted = output.max(1)            
#         print("predicted:", predicted[0])

#         exit(1)
        return output

#         if self.sign_prob6.test_forward:
#             return output
#         else:
#             return output, p       
        
#         if self.sign_prob6.test_forward and self.sign_prob5.test_forward:
#             return output            
#         elif self.sign_prob5.test_forward and not self.sign_prob6.test_forward:
#             return output, p                   
#         else:        
#             return output, p, p5

    def train_mode_switch(self):
        if self.conv1_is_normal is not True:
            self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
        self.sign_prob3.train_mode_switch()
        self.sign_prob4.train_mode_switch()
        self.sign_prob5.train_mode_switch()
        self.sign_prob6.train_mode_switch()
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()    
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.gumble_last_layer:
                self.pool6.train_mode_switch()   
        if self.lrgain:
            self.gain1.train_mode_switch()
            self.gain2.train_mode_switch()
            self.gain3.train_mode_switch()
            self.gain4.train_mode_switch()
            self.gain5.train_mode_switch()
            self.gain6.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        if self.conv1_is_normal is not True:        
            self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.sign_prob1.test_mode_switch(options, tickets)    
        self.sign_prob2.test_mode_switch(options, tickets)
        self.sign_prob3.test_mode_switch(options, tickets)
        self.sign_prob4.test_mode_switch(options, tickets)
        self.sign_prob5.test_mode_switch(options, tickets)
        self.sign_prob6.test_mode_switch(options, tickets)
        if not self.gumble_last_layer:        
            self.fc1.test_mode_switch(options, tickets)      
        if self.pool:
            self.pool2.test_mode_switch(options, tickets)
            self.pool4.test_mode_switch(options, tickets)
            if not self.gumble_last_layer:
                self.pool6.test_mode_switch(options, tickets)  
        if self.lrgain:
            self.gain1.test_mode_switch(options, tickets)
            self.gain2.test_mode_switch(options, tickets)
            self.gain3.test_mode_switch(options, tickets)
            self.gain4.test_mode_switch(options, tickets)
            self.gain5.test_mode_switch(options, tickets)
            self.gain6.test_mode_switch(options, tickets)  

    def just_freeze_layer(self, conv_layer, sign_layer, next_layer):
        print('==> Freezing...')        
        sign_layer.test_mode_switch(1, 1)  
        conv_layer.test_forward = True        
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False         
                
    def freeze_layer(self, conv_layer, sign_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None, update_tau=False, gain_layer=None, pool_layer=None):
        print('==> Freezing...') 
        
#         self.tau = 0.8 * self.tau
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau
        
        if update_tau is True:
            sign_layer.tau = 0.1
            
        if gain_layer is not None:
            gain_layer.test_mode_switch(1, 1)
        
        sign_layer.test_mode_switch(1, 1)
                
        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 

#         for param in sign_layer.bn.parameters():
#             param.requires_grad = False 
            
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)

        net.load_state_dict(torch.load(model_id))
            

    def freeze_last_layer(self, conv_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing Last Layer...') 
        
#         self.tau = 0.1
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau

        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        for param in conv_layer.parameters():
            param.requires_grad = False 
        
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)            
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)          
                
        net.load_state_dict(torch.load(model_id))
            
    def train_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            if self.conv1_is_normal is not True:
                self.conv1.train_mode_switch()
                self.sign_prob1.train_mode_switch()  
                if self.lrgain:                
                    self.gain1.train_mode_switch()                
        if freeze_mask[1] is not 0:
            self.conv2.train_mode_switch()
            self.sign_prob2.train_mode_switch()
            if self.pool:
                self.pool2.train_mode_switch()
            if self.lrgain:                
                self.gain2.train_mode_switch()                   
        if freeze_mask[2] is not 0:
            self.conv3.train_mode_switch()
            self.sign_prob3.train_mode_switch()
            if self.lrgain:                
                self.gain3.train_mode_switch()              
        if freeze_mask[3] is not 0:
            self.conv4.train_mode_switch()
            self.sign_prob4.train_mode_switch()
            if self.pool:
                self.pool4.train_mode_switch()
            if self.lrgain:                
                self.gain4.train_mode_switch()                    
        if freeze_mask[4] is not 0:
            self.conv5.train_mode_switch()
            self.sign_prob5.train_mode_switch()
            if self.lrgain:                
                self.gain5.train_mode_switch()                
        if freeze_mask[5] is not 0:
            self.conv6.train_mode_switch()        
            self.sign_prob6.train_mode_switch()   
            if self.pool:
                self.pool6.train_mode_switch()  
            if self.lrgain:                
                self.gain6.train_mode_switch()                    
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()        
                
    def test_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0:
            if self.conv1_is_normal is not True:            
                self.conv1.test_mode_switch(1, 1)
                self.sign_prob1.test_mode_switch(1, 1)    
                if self.lrgain:                
                    self.gain1.test_mode_switch(1, 1)                    
        if freeze_mask[1] is not 0:
            self.conv2.test_mode_switch(1, 1)
            self.sign_prob2.test_mode_switch(1, 1)
            if self.pool:
                self.pool2.test_mode_switch(1, 1)       
            if self.lrgain:                
                self.gain2.test_mode_switch(1, 1)                    
        if freeze_mask[2] is not 0:
            self.conv3.test_mode_switch(1, 1)
            self.sign_prob3.test_mode_switch(1, 1)
            if self.lrgain:                
                self.gain3.test_mode_switch(1, 1)               
        if freeze_mask[3] is not 0:
            self.conv4.test_mode_switch(1, 1)
            self.sign_prob4.test_mode_switch(1, 1)
            if self.pool:
                self.pool4.test_mode_switch(1, 1)    
            if self.lrgain:                
                self.gain4.test_mode_switch(1, 1)                 
        if freeze_mask[4] is not 0:
            self.conv5.test_mode_switch(1, 1)
            self.sign_prob5.test_mode_switch(1, 1)
            if self.lrgain:                
                self.gain5.test_mode_switch(1, 1)             
        if freeze_mask[5] is not 0:
            self.conv6.test_mode_switch(1, 1)        
            self.sign_prob6.test_mode_switch(1, 1)  
            if self.pool:
                self.pool6.test_mode_switch(1, 1) 
            if self.lrgain:                
                self.gain6.test_mode_switch(1, 1)                 
        if not self.gumble_last_layer:
            self.fc1.test_mode_switch(1, 1)     
#         if self.pool:
#             self.pool2.test_mode_switch(1, 1)
#             self.pool4.test_mode_switch(1, 1)
#             if not self.gumble_last_layer:
#                 self.pool6.test_mode_switch(1, 1)  

    def update_tau(self, sign_layer, new_tau, tau_factor=None, stop_epoch=None, stop_val=0.1):
        if tau_factor is None:
            print('==> updating tau...')         
            sign_layer.tau = new_tau
            sign_layer.hard = True
        else:
            new_val = tau_factor * sign_layer.tau
            if new_val < stop_val:
                sign_layer.tau = sign_layer.tau
            else:
                sign_layer.tau = new_val

# here4
class LRNet_CIFAR10_ver2_l1(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_CIFAR10_ver2_l1, self).__init__()

        self.args = args
        self.sampled_last_layer = False # always False here4
        
        self.bn_s = False
        self.gumbel = True        
        self.gumble_last_layer = False # True if not self.sampled_last_layer else False
        self.bn_last_layer = True or self.bn_s # True if not self.sampled_last_layer else False
        self.gain = True 
        
        self.dropout = False
        self.zero_act = 0
        self.tau = 1.0
        
        self.pool = False
        self.stride = 1 if self.pool else 2
        if self.pool:
            self.pool2 = lrnet_nn.LRnetAvgPool2d()
            self.pool4 = lrnet_nn.LRnetAvgPool2d()
            if self.gumble_last_layer:
                self.pool6 = nn.AvgPool2d(2, stride=2)                
            else:
                self.pool6 = lrnet_nn.LRnetAvgPool2d()
                
        self.fc1_output_chan = 10
        
        self.l1 = True   
        if self.l1:    
            self.conv1 = nn.Conv2d(3, 128, 3, 1, padding=1)
        else:
            self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)

        if self.gumbel:
            self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)          
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_chan=128, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_chan=256, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_chan=512, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        
        self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_chan=512, tau=self.tau, zero_act=self.zero_act, output_sample=self.gumble_last_layer, collect_stats=True, bn_layer = self.bn_last_layer)
 
        if self.gumble_last_layer:
            self.fc1 = nn.Linear(8192, self.fc1_output_chan)
        else:
            self.fc1 = lrnet_nn.LRnetLinear(8192, self.fc1_output_chan)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
    
        if writer is not None:
            self.writer = writer
            self.iteration_train = 0
            self.iteration_test = 0
            self.inner_iteration_test = 0
            self.inner_iteration_train = 0
        self.tensorboard_train = False
        self.tensorboard_test = False    
        
        self.net_id = id_generator(16)
        self.sigmoid_func = torch.nn.Sigmoid()   
#         self.iterations_list = [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
        self.iterations_list = [0, 10, 50, 100, 200, 250, 350, args.epochs-1]
        self.iterations_list1 = [0, 10, 50, 100, 200, 250, 270, 290, 310, 330, 350, args.epochs-1]
                
    def forward(self, x):
        cin = x
        x = self.conv1(x)
        
        if self.l1: 
            x = F.relu(x)            
        else:
            x = self.sign_prob1(x)
            if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
                p = get_p(x, self.sign_prob1.output_sample)
                p = p + 1e-10
                calc_avg_entropy (p, "sign_entropy", "/sign1_entropy", self.iteration_train, self.writer)
                alpha_p = self.sigmoid_func(self.conv1.alpha) + 1e-10            
                betta_p = self.sigmoid_func(self.conv1.betta) * (1 - alpha_p) + 1e-10            
                layer_hist("sign1_p/sign1", p, self.writer, self.iteration_train, self.iterations_list) 
                m_s, v_s = calc_m_v_sample (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                m_a, v_a = calc_m_v_analyt (cin, self.conv1, 2000, 'conv1_clt/conv1', self.writer, self.iteration_train, self.iterations_list)
                compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv1/iteraion", self.writer, self.iteration_train, self.iterations_list)
            x = get_x (x, self.sign_prob1)
        
        cin = x
        x = self.conv2(x)
        if self.pool:
            x = self.pool2(x)
        x = self.sign_prob2(x)
        
        if self.tensorboard_train and self.sign_prob2.collect_stats and not self.sign_prob2.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob2.output_sample)             
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign2_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv2.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv2.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_2", self.inner_iteration_train, self.writer)             
            layer_hist("sign2_p/sign2", p, self.writer, self.iteration_train, self.iterations_list) 
            m_s, v_s = calc_m_v_sample (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
            m_a, v_a = calc_m_v_analyt (cin, self.conv2, 2000, 'conv2_clt/conv2', self.writer, self.iteration_train, self.iterations_list)
            compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv2/iteraion", self.writer, self.iteration_train, self.iterations_list)    
            
        x = get_x (x, self.sign_prob2)   
        cin = x
        x = self.conv3(x)
        x = self.sign_prob3(x)        
        
        if self.tensorboard_train and self.sign_prob3.collect_stats and not self.sign_prob3.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob3.output_sample)                            
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign3_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv3.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv3.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_3", self.inner_iteration_train, self.writer) 
            layer_hist("sign3_p/sign3", p, self.writer, self.iteration_train, self.iterations_list) 
            m_s, v_s = calc_m_v_sample (cin, self.conv3, 2000, 'conv3_clt/conv3', self.writer, self.iteration_train, self.iterations_list)
            m_a, v_a = calc_m_v_analyt (cin, self.conv3, 2000, 'conv3_clt/conv3', self.writer, self.iteration_train, self.iterations_list)
            compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv3/iteraion", self.writer, self.iteration_train, self.iterations_list)    
                        
        x = get_x (x, self.sign_prob3)   
        cin = x        
        x = self.conv4(x)
        if self.pool:
            x = self.pool4(x)        
        x = self.sign_prob4(x)
        
        if self.tensorboard_train and self.sign_prob4.collect_stats and not self.sign_prob4.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob4.output_sample)    
            p = p + 1e-10            
            calc_avg_entropy (p, "sign_entropy", "/sign4_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv4.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv4.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_4", self.inner_iteration_train, self.writer) 
            layer_hist("sign4_p/sign4", p, self.writer, self.iteration_train, self.iterations_list) 
            m_s, v_s = calc_m_v_sample (cin, self.conv4, 2000, 'conv4_clt/conv4', self.writer, self.iteration_train, self.iterations_list)
            m_a, v_a = calc_m_v_analyt (cin, self.conv4, 2000, 'conv4_clt/conv4', self.writer, self.iteration_train, self.iterations_list)
            compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv4/iteraion", self.writer, self.iteration_train, self.iterations_list)    
            
        x = get_x (x, self.sign_prob4)                      
        cin = x
        x = self.conv5(x)
        x = self.sign_prob5(x)
        
        if self.tensorboard_train and self.sign_prob5.collect_stats and not self.sign_prob5.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob5.output_sample)    
            p = p + 1e-10  
            calc_avg_entropy (p, "sign_entropy", "/sign5_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv5.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv5.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_5", self.inner_iteration_train, self.writer)          
            layer_hist("sign5_p/sign5", p, self.writer, self.iteration_train, self.iterations_list) 
            m_s, v_s = calc_m_v_sample (cin, self.conv5, 2000, 'conv5_clt/conv5', self.writer, self.iteration_train, self.iterations_list)
            m_a, v_a = calc_m_v_analyt (cin, self.conv5, 2000, 'conv5_clt/conv5', self.writer, self.iteration_train, self.iterations_list)
            compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv5/iteraion", self.writer, self.iteration_train, self.iterations_list)    
            
        x = get_x (x, self.sign_prob5)                      
        cin = x
        x = self.conv6(x)
        if self.pool:
            x = self.pool6(x)                             
        x = self.sign_prob6(x) 
        
        if self.tensorboard_train and self.sign_prob6.collect_stats and not self.sign_prob6.test_forward and (self.writer is not None): 
            p = get_p(x, self.sign_prob6.output_sample)    
            p = p + 1e-10  
            calc_avg_entropy (p, "sign_entropy", "/sign6_entropy", self.inner_iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv6.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv6.betta) * (1 - alpha_p) + 1e-10            
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_6", self.inner_iteration_train, self.writer)             
            layer_hist("sign6_p/sign6", p, self.writer, self.iteration_train, self.iterations_list)        
            m_s, v_s = calc_m_v_sample (cin, self.conv6, 2000, 'conv6_clt/conv6', self.writer, self.iteration_train, self.iterations_list1)
            m_a, v_a = calc_m_v_analyt (cin, self.conv6, 2000, 'conv6_clt/conv6', self.writer, self.iteration_train, self.iterations_list1)
            compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_conv6/iteraion", self.writer, self.iteration_train, self.iterations_list1)
            
            self.tensorboard_train = False
            self.inner_iteration_test = self.inner_iteration_test + 1
            self.inner_iteration_train = self.inner_iteration_train + 1
#             exit(1)

        if self.sign_prob6.test_forward:
            x = torch.flatten(x, 1)  # 8192            
        elif self.gumble_last_layer:
            if self.sign_prob6.collect_stats:
                z, p = x
                x = torch.flatten(z, 1)  # 8192
            else:
                x = torch.flatten(x, 1)  # 8192
        else: 
            if self.sign_prob6.collect_stats:
                if self.sign_prob6.output_sample:      
                    z, p = x   
                    x = z
                else:
                    m, v, p = x
                    x = m, v                    
            else:
                m, v = x

            if not self.gumble_last_layer:
                m,v = x

            m = torch.flatten(m, 1)  # 8192
            v = torch.flatten(v, 1)  # 8192
            x = m, v
            
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc1(x)  # 8192 -> 10        
        output = x
        
        return output

    def train_mode_switch(self):
        if not self.l1:
            self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()
        if not self.l1:        
            self.sign_prob1.train_mode_switch()
        self.sign_prob2.train_mode_switch()
        self.sign_prob3.train_mode_switch()
        self.sign_prob4.train_mode_switch()
        self.sign_prob5.train_mode_switch()
        self.sign_prob6.train_mode_switch()
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()    
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.gumble_last_layer:
                self.pool6.train_mode_switch()           

    def test_mode_switch(self, options, tickets):
        if not self.l1:                
            self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        if not self.l1:                
            self.sign_prob1.test_mode_switch(options, tickets)    
        self.sign_prob2.test_mode_switch(options, tickets)
        self.sign_prob3.test_mode_switch(options, tickets)
        self.sign_prob4.test_mode_switch(options, tickets)
        self.sign_prob5.test_mode_switch(options, tickets)
        self.sign_prob6.test_mode_switch(options, tickets)
        if not self.gumble_last_layer:        
            self.fc1.test_mode_switch(options, tickets)      
        if self.pool:
            self.pool2.test_mode_switch(options, tickets)
            self.pool4.test_mode_switch(options, tickets)
            if not self.gumble_last_layer:
                self.pool6.test_mode_switch(options, tickets)             

    def freeze_layer(self, conv_layer, sign_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing...') 
        
#         self.tau = 0.8 * self.tau
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau
        
        sign_layer.test_mode_switch(1, 1)
                
        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 

#         for param in sign_layer.bn.parameters():
#             param.requires_grad = False 
            
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)

        net.load_state_dict(torch.load(model_id))

        
    def freeze_l1_layer(self, conv_layer, next_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing...') 
                        
        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        next_layer.sampled_input = True
        for param in conv_layer.parameters():
            param.requires_grad = False 
                                                        

    def freeze_last_layer(self, conv_layer, trials, net, criterion, device, trainloader, args, f=None, pool_layer=None):
        print('==> Freezing Last Layer...') 
        
#         self.tau = 0.1
#         self.sign_prob1.tau = self.tau
#         self.sign_prob2.tau = self.tau
#         self.sign_prob3.tau = self.tau
#         self.sign_prob4.tau = self.tau
#         self.sign_prob5.tau = self.tau
#         self.sign_prob6.tau = self.tau

        if self.pool and pool_layer is not None:
            pool_layer.test_mode_switch(1, 1)
            
        for param in conv_layer.parameters():
            param.requires_grad = False 
        
        best_iter = 0
        best_acc = 0        
        for iteration in range(trials):
            conv_layer.test_mode_switch(1, 1)            
            best_acc, best_iter, iter_acc = test(net, criterion, iteration, device, trainloader, args, best_acc, best_iter, test_mode=False, f=f, eval_mode=True, dont_save=True)
            print('iteration: ', iteration, 'iter_acc: ', iter_acc, 'best_iter: ', best_iter, 'best_acc: ', best_acc)
            if (iter_acc > best_acc) or iteration is 0:
                model_id = 'saved_models/freezed_model_' + str(self.net_id) + '.pt'
                print('==> saving model:', model_id)
                torch.save(net.state_dict(), model_id)          
                
        net.load_state_dict(torch.load(model_id))
            
    def train_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0 and not self.l1:
            self.conv1.train_mode_switch()
            self.sign_prob1.train_mode_switch()    
        if freeze_mask[1] is not 0:
            self.conv2.train_mode_switch()
            self.sign_prob2.train_mode_switch()
        if freeze_mask[2] is not 0:
            self.conv3.train_mode_switch()
            self.sign_prob3.train_mode_switch()
        if freeze_mask[3] is not 0:
            self.conv4.train_mode_switch()
            self.sign_prob4.train_mode_switch()
        if freeze_mask[4] is not 0:
            self.conv5.train_mode_switch()
            self.sign_prob5.train_mode_switch()
        if freeze_mask[5] is not 0:
            self.conv6.train_mode_switch()        
            self.sign_prob6.train_mode_switch()   
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()        
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.gumble_last_layer:
                self.pool6.train_mode_switch()      
                
    def test_mode_freeze(self, freeze_mask):
        if freeze_mask[0] is not 0 and not self.l1:
            self.conv1.test_mode_switch(1, 1)
            self.sign_prob1.test_mode_switch(1, 1)    
        if freeze_mask[1] is not 0:
            self.conv2.test_mode_switch(1, 1)
            self.sign_prob2.test_mode_switch(1, 1)
        if freeze_mask[2] is not 0:
            self.conv3.test_mode_switch(1, 1)
            self.sign_prob3.test_mode_switch(1, 1)
        if freeze_mask[3] is not 0:
            self.conv4.test_mode_switch(1, 1)
            self.sign_prob4.test_mode_switch(1, 1)
        if freeze_mask[4] is not 0:
            self.conv5.test_mode_switch(1, 1)
            self.sign_prob5.test_mode_switch(1, 1)
        if freeze_mask[5] is not 0:
            self.conv6.test_mode_switch(1, 1)        
            self.sign_prob6.test_mode_switch(1, 1)   
        if not self.gumble_last_layer:
            self.fc1.test_mode_switch(1, 1)     
        if self.pool:
            self.pool2.test_mode_switch(1, 1)
            self.pool4.test_mode_switch(1, 1)
            if not self.gumble_last_layer:
                self.pool6.test_mode_switch(1, 1)  







