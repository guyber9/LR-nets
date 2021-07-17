import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.nn.modules.conv import _single, _pair, _triple, _reverse_repeat_tuple
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from utils import mean_over_channel, print_full_tensor, print_fullllll_tensor

class LRnetConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        clusters: int = 3,
        transposed: bool = True,
        test_forward: bool = False,
        output_sample: bool = True,
        binary_mode: bool = False,
    ):
        super(LRnetConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward, self.output_sample, self.binary_mode = test_forward, output_sample, binary_mode
        self.transposed = transposed
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32

        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))

        discrete_prob = np.array([-1.0, 0.0, 1.0])
        discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
        self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensor_dtype, device=self.device)
        self.discrete_square_mat = self.discrete_mat * self.discrete_mat

        self.num_of_options = 30
        self.test_weight_arr = []
        self.cntr = 0
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, -0.69314)
        init.constant_(self.betta, 0.0)
        # init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        # init.kaiming_uniform_(self.betta, a=math.sqrt(5))
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))

    def train_mode_switch(self) -> None:
        # print ("train_mode_switch")
        self.test_forward = False

    def test_mode_switch(self, num_of_options=1, tickets=10) -> None:
        # print ("test_mode_switch")
        self.test_forward = True
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
        # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
        # self.test_weight = torch.tensor(sampled, dtype=self.tensor_dtype, device=self.device)

        self.num_of_options = num_of_options
        self.test_weight_arr = []
        if tickets > 1:
            m = torch.distributions.Multinomial(tickets, prob_mat)
        else:
            m = torch.distributions.Categorical(prob_mat)
        for idx in range(0, self.num_of_options):
            sampled = m.sample()
            if tickets > 1:
                values = torch.argmax(sampled, dim=4) - 1
            else:
                values = sampled - 1
            self.test_weight_arr.append(values)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     print ("alpha: " + str(self.alpha))
            #     print ("betta: " + str(self.betta))
            #     print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
            #     print("betta isnan: " + str(torch.isnan(self.betta).any()))
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean
            sigma_square = mean_square - mean_pow2

            # if torch.cuda.is_available():
            #     torch.backends.cudnn.deterministic = True

            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     file2 = {'w': sigma_square}
            #     torch.save(file2, 'my_tensors2.pt')
            #     w = sigma_square
            #     print("sigma_square bfr conv: " + str(sigma_square))

            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            z1 = torch.relu(z1) ##TODO

            # if torch.cuda.is_available():
            #     torch.backends.cudnn.deterministic = False

            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     print("sigma_square size: " + str(sigma_square.size()))
            #     print("sigma_square: " + str(sigma_square))
            #     file2 = {'w': sigma_square}
            #     torch.save(file2, 'my_tensors2.pt')
            #     print_fullllll_tensor(sigma_square, "sigma_square")
            #     print_full_tensor(z1, "z1")
            #     print_full_tensor(torch.relu(z1), "relu(z1)")
            #     print("sigma_square isnan: " + str(torch.isnan(sigma_square).any()))
            #     print("z1 isnan: " + str(torch.isnan(z1).any()))

            v = torch.sqrt(z1)

            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     if torch.isnan(v).any():
            #         print("v isnan: " + str(torch.isnan(v).any()))
            #         torch.set_printoptions(threshold=10_000)
            #         # print ("input^2: " + str(input * input))
            #         # print("sigma_square: " + str(sigma_square))
            #         x = input*input
            #         print("sigma_square: " + str(sigma_square))
            #         print("x: " + str(x))
            #         m = {'x': x, 'w': sigma_square}
            #         # file1 = {'x': x}
            #         # file2 = {'w': sigma_square}
            #         # torch.save(file1, 'my_tensors1.pt')
            #         # torch.save(file2, 'my_tensors2.pt')
            #         torch.save(m, 'my_tensors.pt')
            #         exit(1)

            #     print("v: " + str(v))
            #     print("m isnan: " + str(torch.isnan(m).any()))

            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     print_full_tensor(v, "v")
            #     exit(1)

            if self.output_sample:
                epsilon = torch.rand(z1.size(), requires_grad=False, dtype=self.tensor_dtype, device=self.device)
                return m + epsilon * v
            else:
                return m, v


# class LRnetConv2d_not_sample(nn.Module):
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: _size_2_t,
#         stride: _size_2_t = 1,
#         padding: _size_2_t = 0,
#         dilation: _size_2_t = 1,
#         groups: int = 1,
#         clusters: int = 3,
#         transposed: bool = True,
#         test_forward: bool = False,
#     ):
#         super(LRnetConv2d_not_sample, self).__init__()
#         self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
#         self.test_forward = test_forward
#         self.transposed = transposed
#         if torch.cuda.is_available():
#             self.device = 'cuda'
#         else:
#             self.device = 'cpu'
#         self.tensor_dtype = torch.float32
#
#         if self.transposed:
#             D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
#         else:
#             D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size
#
#         self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
#         self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
#         self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
#         self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))
#
#         discrete_prob = np.array([-1.0, 0.0, 1.0])
#         discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
#         self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensor_dtype, device=self.device)
#         self.discrete_square_mat = self.discrete_mat * self.discrete_mat
#
#         self.num_of_options = 30
#         self.test_weight_arr = []
#         self.cntr = 0
#         self.sigmoid = torch.nn.Sigmoid()
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         # init.constant_(self.alpha, -0.69314)
#         # init.constant_(self.betta, 0.0)
#         # init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
#         # init.kaiming_uniform_(self.betta, a=math.sqrt(5))
#         torch.nn.init.normal_(self.alpha, mean=0.0, std=0.01)
#         torch.nn.init.normal_(self.betta, mean=0.0, std=0.01)
#         if self.bias is not None:
#             prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#
#     def initialize_weights(self, alpha, betta) -> None:
#         print ("Initialize Weights")
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
#         self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))
#
#     def train_mode_switch(self) -> None:
#         # print ("train_mode_switch")
#         self.test_forward = False
#
#     def test_mode_switch(self, num_of_options=1, tickets=10) -> None:
#         # print ("test_mode_switch")
#         self.test_forward = True
#         sigmoid_func = torch.nn.Sigmoid()
#         alpha_prob = sigmoid_func(self.alpha)
#         betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
#         prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
#         # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
#         # self.test_weight = torch.tensor(sampled, dtype=self.tensor_dtype, device=self.device)
#
#         self.num_of_options = num_of_options
#         self.test_weight_arr = []
#         if tickets > 1:
#             m = torch.distributions.Multinomial(tickets, prob_mat)
#         else:
#             m = torch.distributions.Categorical(prob_mat)
#         for idx in range(0, self.num_of_options):
#             sampled = m.sample()
#             if tickets > 1:
#                 values = torch.argmax(sampled, dim=4) - 1
#             else:
#                 values = sampled - 1
#             self.test_weight_arr.append(values)
#
#     def forward(self, input: Tensor) -> Tensor:
#         if self.test_forward:
#             self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
#             return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         else:
#             prob_alpha = self.sigmoid(self.alpha)
#             prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
#             prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
#             # E[X] calc
#             mean_tmp = prob_mat * self.discrete_mat
#             mean = torch.sum(mean_tmp, dim=4)
#             m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups) # TODO bias
#             # E[x^2]
#             mean_square_tmp = prob_mat * self.discrete_square_mat
#             mean_square = torch.sum(mean_square_tmp, dim=4)
#             # E[x] ^ 2
#             mean_pow2 = mean * mean
#             sigma_square = mean_square - mean_pow2
#             if torch.cuda.is_available():
#                 torch.backends.cudnn.deterministic = True
#             z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
#             if torch.cuda.is_available():
#                 torch.backends.cudnn.deterministic = False
#             v = torch.sqrt(z1)
#
#             # print ("m: " + str(m))
#             # print ("v: " + str(v))
#
#             return m, v

class LRnetConv2d_ver2(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        clusters: int = 3,
        transposed: bool = True,
        test_forward: bool = False,
        output_sample: bool = False,
    ):
        super(LRnetConv2d_ver2, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
        self.transposed = transposed
        self.output_sample = output_sample
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32

        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))

        discrete_prob = np.array([-1.0, 0.0, 1.0])
        discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
        self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensor_dtype, device=self.device)
        self.discrete_square_mat = self.discrete_mat * self.discrete_mat

        self.num_of_options = 30
        self.test_weight_arr = []
        self.cntr = 0
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.constant_(self.alpha, -0.69314)
        # init.constant_(self.betta, 0.0)
        torch.nn.init.normal_(self.alpha, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.betta, mean=0.0, std=0.01)
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))

    def train_mode_switch(self) -> None:
        # print ("train_mode_switch")
        self.test_forward = False

    def test_mode_switch(self, num_of_options, tickets=10) -> None:
        self.test_forward = True
        # print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)

        # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
        # self.test_weight = torch.tensor(sampled, dtype=self.tensor_dtype, device=self.device)

        self.num_of_options = num_of_options
        self.test_weight_arr = []
        if tickets > 1:
            m = torch.distributions.Multinomial(tickets, prob_mat)
        else:
            m = torch.distributions.Categorical(prob_mat)
        for idx in range(0, self.num_of_options):
            sampled = m.sample()
            if tickets > 1:
                values = torch.argmax(sampled, dim=4) - 1
            else:
                values = sampled - 1
            self.test_weight_arr.append(values)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            sign_input = torch.sign(input)
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
            return F.conv2d(sign_input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            m, v = input
            # print ("m: " + str(m.size()))
            # print ("v: " + str(v.size()))
            # print ("m: " + str(m))
            # print ("v: " + str(v))

            # print ("alpha: " + str(self.alpha))
            # print ("betta: " + str(self.betta))
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)

            # mean of input
            input_mean = 2 * (1 - torch.erf((-1) * m / v)) - 1
            # print("input_mean: " + str(input_mean))

            m1 = F.conv2d(input_mean, mean, self.bias, self.stride, self.padding, self.dilation, self.groups) # TODO bias
            # print("m1: " + str(m1))
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean

            # sigma_square = mean_square - mean_pow2
            e_h_2 = torch.ones(m.size(), dtype=self.tensor_dtype, device=self.device)

            # print("mean_square_tmp: " + str(mean_square_tmp))
            # print("mean_square: " + str(mean_square))
            # print("e_h_2: " + str(e_h_2))

            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True

            # z1 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)
            # z1 = m1 * m1
            z2 = F.conv2d(e_h_2, mean_square, None, self.stride, self.padding, self.dilation, self.groups)
            z3 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)

            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False

            # print("z2: " + str(z2))
            # print("z3: " + str(z3))

            # z = z1 + z2 - z3
            z = z2 - z3
            z = torch.relu(z) # TODO
            v1 = torch.sqrt(z)

            # print ("m: " + str(m))
            # print ("v: " + str(v))

            # print("z: " + str(z))
            # exit(1)

            # print("m1 isnan: " + str(torch.isnan(m1).any()))
            # print("v1 isnan: " + str(torch.isnan(v1).any()))
            # print("z is negative: " + str((z < 0).any()))

            if self.output_sample:
                epsilon = torch.rand(z.size(), requires_grad=False, dtype=self.tensor_dtype, device=self.device)
                return m1 + epsilon * v1
            else:
                return m1, v1


class LRBatchNorm2d(nn.Module):

    def __init__(
        self,
        channels: int,
        eps: int = 1e-05,
        test_forward: bool = False
    ):
        super(LRBatchNorm2d, self).__init__()
        self.channels, self.eps, self.test_forward = channels, eps, test_forward
        self.test_forward = test_forward
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32

        self.weight = torch.nn.Parameter(torch.ones([self.channels], dtype=torch.float32, device=self.device))
        self.bias = torch.nn.Parameter(torch.zeros([self.channels], dtype=self.tensor_dtype, device=self.device))

        self.bn = nn.BatchNorm2d(self.channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self) -> None:
        self.test_forward = True

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return self.bn(input)
        else:
            m, v = input

            # print("#################################")
            # print("m: \n" + str(m))
            # print("#################################")
            # print("mean:")
            mean = mean_over_channel(m)
            # print("#################################")
            # print("mean_square:")
            mean_square = mean_over_channel(m * m)
            # print("#################################")
            # print("sigma_square:")
            sigma_square = mean_over_channel(v * v)

            weights_tmp = self.weight.repeat(m.size(0), 1)
            iweights = weights_tmp.view(m.size(0), m.size(1), 1, 1)
            bias_tmp = self.weight.repeat(m.size(0), 1)
            ibias = bias_tmp.view(m.size(0), m.size(1), 1, 1)

            # print("mean size: " + str(mean.size()))
            # print("mean_square size: " + str(mean_square.size()))
            # print("sigma_square size: " + str(sigma_square.size()))
            # print("iweights size: " + str(iweights.size()))
            # print("ibias size: " + str(ibias.size()))

            variance = sigma_square + mean_square - (mean * mean) + self.eps
            std = torch.sqrt(variance)

            norm_m = ((m - mean) / std)
            norm_v = (v / std)
            # norm_m = (iweights * ((m - mean) / std)) + ibias
            # norm_v = iweights * (v / std)

            if torch.isnan(mean).any():
                print("m isnan: " + str(torch.isnan(m).any()))
                print("v isnan: " + str(torch.isnan(v).any()))
                print("mean isnan: " + str(torch.isnan(mean).any()))
                print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                exit(1)

            if torch.isnan(variance).any():
                print("m isnan: " + str(torch.isnan(m).any()))
                print("v isnan: " + str(torch.isnan(v).any()))
                print("variance isnan: " + str(torch.isnan(variance).any()))
                print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                exit(1)

            if torch.isnan(std).any():
                print("variance: \n" + str(variance))
                print("variance is negative: " + str((variance < 0).any()))
                print("m isnan: " + str(torch.isnan(m).any()))
                print("v isnan: " + str(torch.isnan(v).any()))
                print("variance isnan: " + str(torch.isnan(variance).any()))
                print("std isnan: " + str(torch.isnan(std).any()))
                print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                exit(1)

            # print("m size: " + str(m.size()))
            # print("v size: " + str(v.size()))
            #
            # mean = mean_over_channel(m)
            # mean_square = mean_over_channel(m * m)
            # sigma_square = mean_over_channel(v * v)
            # variance = sigma_square + mean_square - (mean * mean) + self.eps
            # std = torch.sqrt(variance)
            #
            # print("mean size: " + str(mean.size()))
            # print("new_mean_square size: " + str(mean_square.size()))
            # print("new_sigma_square size: " + str(sigma_square.size()))
            # print("variance size: " + str(variance.size()))
            # print("std size: " + str(std.size()))
            #
            #
            # exit(1)
            #
            # # print("m: " + str(m))
            # # print("v: " + str(v))
            #
            # # epsilon = torch.rand(v.size(), requires_grad=False, dtype=self.tensor_dtype, device=self.device)
            # # sampled_input = m + epsilon * v
            #
            # mean = torch.mean(m)
            # # print("mean: " + str(mean))
            # mean_square = torch.mean(m * m)
            # # print("mean_square: " + str(mean_square))
            # sigma_square = torch.mean(v * v)
            # # print("sigma_square: " + str(sigma_square))
            # variance = sigma_square + mean_square - (mean * mean) + self.eps
            # # print("variance: " + str(variance))
            # std = torch.sqrt(variance)
            # # print("std: " + str(std))
            #
            # # norm_m = (self.weight * ((m - mean) / std)) + self.bias
            # # norm_v = self.weight * (v / std)
            #
            # norm_m = (m - mean) / std
            # norm_v = v / std
            #
            # # exit(1)
            #
            # # weight = torch.unsqueeze(self.weight, 1)
            # # weight = weight.repeat(1, 16)
            # # weight = torch.reshape(weight, (self.channels, 4, 4))
            #
            # # print("norm_m: " + str(norm_m))
            # # print("norm_v: " + str(norm_v))
            #
            # # print("mean of norm_m: " + str(torch.mean(norm_m)))
            # # print("var of norm_m: " + str(torch.var(norm_m)))
            # # epsilon1 = torch.rand(v.size(), requires_grad=False, dtype=self.tensor_dtype, device=self.device)
            # # sampled_output = norm_m + epsilon1 * norm_v
            # # print("\n\nmean of sampled_input: " + str(torch.mean(sampled_input)))
            # # print("var of sampled_input: " + str(torch.var(sampled_input)))
            # # print("mean of sampled_output: " + str(torch.mean(sampled_output)))
            # # print("var of sampled_output: " + str(torch.var(sampled_output)))
            # # exit(1)

            return norm_m, norm_v
