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
from utils import print_full_tensor, assertnan, collect_hist, collect_m_v, take_silce
import numpy as np

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

##################
## CIFAR10 nets ##
##################

class FPNet_CIFAR10(nn.Module):

    def __init__(self):
        super(FPNet_CIFAR10, self).__init__()
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
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # 1024 -> 10
        output = x
        return output

class LRNet_CIFAR10(nn.Module):

    def __init__(self):
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
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.25) # 0.2 was 93.13
        self.dropout4 = nn.Dropout(0.4) # 0.2 was 93.13
        self.dropout5 = nn.Dropout(0.25) # 0.2 was 93.13
        self.dropout6 = nn.Dropout(0.4) # 0.2 was 93.13
        self.dropout7 = nn.Dropout(0.25) # 0.2 was 93.13
        
    def forward(self, x): 
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)  # <- problematic batchnoram (?)
        x = F.relu(x)
#         x = self.dropout3(x)
        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
#         x = self.dropout5(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
#         x = self.dropout7(x)
        x = self.conv6(x)  # 512 x 8 x 8
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
        self.conv3 = lrnet_nn.LRnetConv2d_ver2(128, 256, 3, stride=1, padding=1, output_sample=False)
        self.conv4 = lrnet_nn.LRnetConv2d_ver2(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv5 = lrnet_nn.LRnetConv2d_ver2(256, 512, 3, stride=1, padding=1, output_sample=False)
        self.conv6 = lrnet_nn.LRnetConv2d_ver2(512, 512, 3, stride=2, padding=1, output_sample=False)
        self.sign_prob = lrnet_nn.LRnet_sign_prob()
        self.bn1 = lrnet_nn.LRBatchNorm2d(128, affine=False)
        self.bn2 = lrnet_nn.LRBatchNorm2d(128, affine=False)
        self.bn3 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn4 = lrnet_nn.LRBatchNorm2d(256, affine=False)
        self.bn5 = lrnet_nn.LRBatchNorm2d(512, affine=False)
        self.bn6 = lrnet_nn.LRBatchNorm2d(512, affine=False)
#         self.bn6 = nn.BatchNorm2d(512)
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
#         self.fc1 = nn.Linear(8192, 1024)
        self.fc1 = lrnet_nn.LRnetLinear(8192, 1024)
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

#         test_samples = []
#         with torch.no_grad():
#             for i in range(1000):
#                 print (i)
#                 # rand weight
#                 self.conv1.test_mode_switch(1,1)
#                 # rand input
# #                     epsilon = torch.normal(0, 1, size=x_m.size())
# #                     sampled_input = m + epsilon * v 
#                 # calc output
#                 y1 = self.conv1(net_input)
#                 test_samples.append(y1.data.cpu().numpy())   
# #                 test_samples.append(y1.data)   
#         test_samples = numpy.asarray(test_samples)
#         test_samples = np.concatenate(test_samples,axis=3).reshape(-1,N)
#         test_mean_s = test_samples.mean(axis=1)
#         test_std_s = test_samples.std(axis=1)            
#         layer_name = 'conv1'
#         with open('layers/' + str(layer_name) + '_test_m.npy', 'wb') as f:
#             np.save(f, test_mean_s)
#         with open('layers/' + str(layer_name) + '_test_v.npy', 'wb') as f:
#             np.save(f, test_std_s)
#         exit(1)
                
        if self.tensorboard_train and (self.writer is not None):     
            m,v=x  

            with torch.no_grad():            
                self.conv1.test_mode_switch()
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv1.device)
                train_x = m + epsilon * v
                test_x = self.conv1(net_input)
                self.conv1.train_mode_switch()
            self.writer.add_scalar("conv1 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)     
                            
            take_silce('layer1_full_output', x)    
                
            collect_m_v(self.writer, 1, x, self.iteration_train)
            self.writer.add_scalar("m1 [0,0,0] mean", torch.mean(m[:,0,0,0]), self.iteration_train)     
            self.writer.add_scalar("v1 [0,0,0] mean", torch.mean(v[:,0,0,0]), self.iteration_train)     
            if (self.iteration_train == 0):
                self.writer.add_histogram("m1 [0,0,0] iteration" + str(0) + " distribution", m[:,0,0,0], 0)              
                self.writer.add_histogram("v1 [0,0,0] iteration" + str(0) + " distribution", v[:,0,0,0], 0) 
            if (self.iteration_train == self.train_last_epoch):
                self.writer.add_histogram("m1 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", m[:,0,0,0], 0)              
                self.writer.add_histogram("v1 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", v[:,0,0,0], 0)                  
                
        if self.tensorboard_test and (self.writer is not None):     
            self.writer.add_scalar("x1 mean", torch.mean(x), self.iteration_test)    
            self.writer.add_scalar("x1 std", torch.std(x), self.iteration_test) 
            collect_hist(0, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
#             collect_hist(50, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
#             collect_hist(100, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)
            collect_hist(self.test_last_epoch, self.iteration_test, self.writer, x, 'x1', self.conv1.test_weight, self.conv1.alpha, self.conv1.betta)         
            
            self.writer.add_scalar("x1 [0,0,0] mean", torch.mean(x[:,0,0,0]), self.iteration_test)     
            self.writer.add_scalar("x1 [0,0,0] std", torch.std(x[:,0,0,0]), self.iteration_test)                   
            if (self.iteration_test == 0):
                self.writer.add_histogram("x1 [0,0,0] iteration" + str(0) + " distribution", x[:,0,0,0], 0)  
            if (self.iteration_test == self.test_last_epoch):
                self.writer.add_histogram("x1 [0,0,0] iteration" + str(self.test_last_epoch) + " distribution", x[:,0,0,0], 0)  

#             if (self.iteration%3 == 0):                 
#                 self.writer.add_histogram('alpha1 distribution', self.conv1.alpha + self.iteration, self.iteration)
#         x = self.bn1(x)
        

        if self.tensorboard_train and (self.writer is not None):      
            with torch.no_grad():            
                m, v = x
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv2.device)
                net_input = m + epsilon * v
    
        x = self.conv2(x)  # 128 x 32 x 32

        if self.tensorboard_train and (self.writer is not None):                            
            take_silce('layer2_full_output', x)    
#         x = self.bn2(x)
        # x = F.relu(x)

        if self.tensorboard_train and (self.writer is not None):      
            m, v = x
            with torch.no_grad():            
                self.conv2.test_mode_switch()
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv2.device)
                train_x = m + epsilon * v
                test_x = self.conv2(net_input)
                self.conv2.train_mode_switch()
            self.writer.add_scalar("conv2 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)        

        if self.tensorboard_train and (self.writer is not None):      
            with torch.no_grad():            
                m, v = x
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv3.device)
                net_input = m + epsilon * v
                
        x = self.conv3(x)  # 256 x 16 x 16

        if self.tensorboard_train and (self.writer is not None):      
            m, v = x
            with torch.no_grad():            
                self.conv3.test_mode_switch()
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv3.device)
                train_x = m + epsilon * v
                test_x = self.conv3(net_input)
                self.conv3.train_mode_switch()
            self.writer.add_scalar("conv3 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)                       
        
        if self.tensorboard_train and (self.writer is not None):                            
            take_silce('layer3_full_output', x)        
#         x = self.bn3(x)
        # x = F.relu(x)
        x = self.conv4(x)  # 256 x 16 x 16
        if self.tensorboard_train and (self.writer is not None):                            
            take_silce('layer4_full_output', x)         
#         x = self.bn4(x)
        # x = F.relu(x)        
        x = self.conv5(x)  # 512 x 8 x 8
        if self.tensorboard_train and (self.writer is not None):     
            take_silce('layer5_full_output', x)
#         x = self.bn5(x)
        # x = F.relu(x)
        
        if self.tensorboard_train and (self.writer is not None):      
            with torch.no_grad():            
                m, v = x
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv6.device)
                net_input = m + epsilon * v

        x = self.conv6(x)  # 512 x 8 x 8
        if self.tensorboard_train and (self.writer is not None):      
            m, v = x
            with torch.no_grad():            
                self.conv6.test_mode_switch()
                epsilon = torch.normal(0, 1, size=m.size(), requires_grad=False, device=self.conv6.device)
                train_x = m + epsilon * v
                test_x = self.conv6(net_input)
                self.conv6.train_mode_switch()
            self.writer.add_scalar("conv6 SAD", torch.mean(torch.abs(train_x - test_x)), self.iteration_train)                
            
            take_silce('layer6_full_output', x)   
#             exit(1)
#         if self.tensorboard_train and (self.writer is not None):     
#             m,v=x
#             collect_m_v(self.writer, 6, x, self.iteration_train)
#             self.writer.add_scalar("m6 [0,0,0] mean", torch.mean(m[:,0,0,0]), self.iteration_train)     
#             self.writer.add_scalar("v6 [0,0,0] mean", torch.mean(v[:,0,0,0]), self.iteration_train)     
#             if (self.iteration_train == 0):
#                 self.writer.add_histogram("m6 [0,0,0] iteration" + str(0) + " distribution", m[:,0,0,0], 0)              
#                 self.writer.add_histogram("v6 [0,0,0] iteration" + str(0) + " distribution", v[:,0,0,0], 0)    
#             if (self.iteration_train == self.train_last_epoch):
#                 self.writer.add_histogram("m6 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", m[:,0,0,0], 0)              
#                 self.writer.add_histogram("v6 [0,0,0] iteration" + str(self.train_last_epoch) + " distribution", v[:,0,0,0], 0)                 
#         if self.tensorboard_test and (self.writer is not None):     
#             collect_hist(0, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
#             collect_hist(50, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
#             collect_hist(100, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
#             collect_hist(self.test_last_epoch, self.iteration_test, self.writer, x, 'x6', self.conv6.test_weight, self.conv6.alpha, self.conv6.betta)
# 
#             self.writer.add_scalar("x6 [0,0,0] mean", torch.mean(x[:,0,0,0]), self.iteration_test)
#             self.writer.add_scalar("x6 [0,0,0] std", torch.std(x[:,0,0,0]), self.iteration_test)  
#             if (self.iteration_test == 0):
#                 self.writer.add_histogram("x6 [0,0,0] iteration" + str(0) + " distribution", x[:,0,0,0], 0)  
#             if (self.iteration_test == self.test_last_epoch):
#                 self.writer.add_histogram("x6 [0,0,0] iteration" + str(self.test_last_epoch) + " distribution", x[:,0,0,0], 0)              
#         x = self.bn6(x)
        x = self.sign_prob(x)        
#         x = F.relu(x)
                   
        if self.fc1.test_forward:
            x = torch.flatten(x, 1)  # 8192
        else: 
            m, v = x
            m = torch.flatten(m, 1)  # 8192
            v = torch.flatten(v, 1)  # 8192
            x = m, v
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
#         x = self.dropout1(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        
        if self.tensorboard_test and (self.writer is not None):
            self.iteration_test = self.iteration_test + 1        
        if self.tensorboard_train and (self.writer is not None):
            self.iteration_train = self.iteration_train + 1        

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
        self.bn3.train_mode_switch()
        self.bn4.train_mode_switch()
        self.bn5.train_mode_switch()
        self.bn6.train_mode_switch()
        self.fc1.train_mode_switch()        

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
        self.bn3.test_mode_switch()
        self.bn4.test_mode_switch()
        self.bn5.test_mode_switch()
        self.bn6.test_mode_switch()
        self.fc1.test_mode_switch(options, tickets)        

    def use_batch_stats_switch(self, new_val):
        self.bn1.use_batch_stats_switch(new_val)
        self.bn2.use_batch_stats_switch(new_val)
        self.bn3.use_batch_stats_switch(new_val)
        self.bn4.use_batch_stats_switch(new_val)
        self.bn5.use_batch_stats_switch(new_val)
        self.bn6.use_batch_stats_switch(new_val)

    def collect_stats_switch(self, new_val):
        self.bn1.collect_stats_switch(new_val)
        self.bn2.collect_stats_switch(new_val)
        self.bn3.collect_stats_switch(new_val)
        self.bn4.collect_stats_switch(new_val)
        self.bn5.collect_stats_switch(new_val)
        self.bn6.collect_stats_switch(new_val)
        
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
