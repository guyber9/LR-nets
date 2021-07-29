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
from utils import print_full_tensor, assertnan

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


class LRNet_ver2(nn.Module):

    def __init__(self):
        super(LRNet_ver2, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 2 ,output_sample=False)
        # self.conv2 = lrnet_nn.LRnetConv2d_ver2(32, 32, 5, 1, output_sample=False)
        self.conv3 = lrnet_nn.LRnetConv2d_ver2(32, 64, 5, 1, output_sample=True)

        # self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 1)
        # self.conv2 = lrnet_nn.LRnetConv2d(32, 32, 5, 1)
        # self.conv3 = lrnet_nn.LRnetConv2d(32, 64, 5, 1)

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(6400, 512)
        # self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 10)

        self.bn1 = lrnet_nn.LRBatchNorm2d(32)
        # self.bn2 = lrnet_nn.LRBatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        m,v = x
        print("m 1size: " + str(m.size()))
        # print("m1: ", m)
        # print("v1: ", v)
        # assertnan(m, "m1")
        # assertnan(v, "v1")
        x = self.bn1(x)
        m,v = x
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
        x = self.conv3(x)  # 64 x 16 x 16 / 64 x 20 x 20
        m,v = x
        print("m2 size: " + str(m.size()))
        assertnan(x, "x3")
        x = self.bn3(x)
        assertnan(x, "bn3")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 64 x 8 x 8 / # 64 x 10 x 10
        print("x size: " + str(x.size()))
        x = torch.flatten(x, 1)  # 1024
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.bn1.train_mode_switch()
        self.bn2.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.bn1.test_mode_switch()
        self.bn2.test_mode_switch()

    def inc_cntr(self):
        self.conv1.cntr = self.conv1.cntr + 1
        self.conv2.cntr = self.conv2.cntr + 1
        self.conv3.cntr = self.conv3.cntr + 1

    def rst_cntr(self):
        self.conv1.cntr = 0
        self.conv2.cntr = 0
        self.conv3.cntr = 0

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
        self.dropout3 = nn.Dropout(0.2) # 0.2 was 93.13
        self.dropout4 = nn.Dropout(0.2) # 0.2 was 93.13
        self.dropout5 = nn.Dropout(0.2) # 0.2 was 93.13
        self.dropout6 = nn.Dropout(0.2) # 0.2 was 93.13
        self.dropout7 = nn.Dropout(0.2) # 0.2 was 93.13
        
    def forward(self, x): 
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)  # <- problematic batchnoram (?)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout7(x)
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

    def __init__(self):
        super(LRNet_CIFAR10_ver2, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, stride=1, padding=1, output_sample=False)
        self.conv2 = lrnet_nn.LRnetConv2d_ver2(128, 128, 3, stride=2, padding=1, output_sample=False)
        self.conv3 = lrnet_nn.LRnetConv2d_ver2(128, 256, 3, stride=1, padding=1, output_sample=False)
        self.conv4 = lrnet_nn.LRnetConv2d_ver2(256, 256, 3, stride=2, padding=1, output_sample=False)
        self.conv5 = lrnet_nn.LRnetConv2d_ver2(256, 512, 3, stride=1, padding=1, output_sample=False)
        self.conv6 = lrnet_nn.LRnetConv2d_ver2(512, 512, 3, stride=2, padding=1, output_sample=True)
        self.bn1 = lrnet_nn.LRBatchNorm2d(128)
        self.bn2 = lrnet_nn.LRBatchNorm2d(128)
        self.bn3 = lrnet_nn.LRBatchNorm2d(256)
        self.bn4 = lrnet_nn.LRBatchNorm2d(256)
        self.bn5 = lrnet_nn.LRBatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        # self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)
        # x = F.relu(x)
        # imax = torch.max(x)
        # imin = torch.min(x)
        # print("hist1 max: " + str(imax))
        # print("hist1 min: " + str(imin))
        # hist1 = torch.histc(x, bins=100, min=-20, max=20)
        # torch.set_printoptions(threshold=10_000)
        # print(hist1)

        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        # x = F.relu(x)
        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        # x = F.relu(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        # x = F.relu(x)
        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        # x = F.relu(x)
        x = self.conv6(x)  # 512 x 8 x 8
        x = self.bn6(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)  # 8192
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
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
        self.bn1.train_mode_switch()
        self.bn2.train_mode_switch()
        self.bn3.train_mode_switch()
        self.bn4.train_mode_switch()
        self.bn5.train_mode_switch()
        # self.bn6.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)
        self.bn1.test_mode_switch()
        self.bn2.test_mode_switch()
        self.bn3.test_mode_switch()
        self.bn4.test_mode_switch()
        self.bn5.test_mode_switch()
        # self.bn6.test_mode_switch()

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
