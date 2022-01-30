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
from utils import mean_over_channel, print_full_tensor, print_fullllll_tensor, print_neg_val, compare_m_v, calc_m_v_sample, calc_m_v_analyt, calc_batch_m_v

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
        output_gain : bool = False, 
        eps: int = 1e-07,
    ):
        super(LRnetConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward, self.output_sample, self.binary_mode = test_forward, output_sample, binary_mode
        self.transposed = transposed
        self.eps = eps
        self.output_gain = output_gain        
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
        self.test_weight = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3], dtype=self.tensor_dtype, device=self.device))
        self.test_weight.requires_grad = False
        
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))
#         self.bias = None
        
#         self.weight = torch.nn.Parameter(torch.ones([out_channels], dtype=self.tensor_dtype, device=self.device))
        
        discrete_prob = np.array([-1.0, 0.0, 1.0])
        discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
        self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensor_dtype, device=self.device)
        self.discrete_square_mat = self.discrete_mat * self.discrete_mat

        if self.output_gain:
            self.gain = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))        
        
        # self.num_of_options = 30
        # self.test_weight_arr = []
        self.cntr = 0
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
#         init.constant_(self.alpha, -0.69314)
#         init.constant_(self.betta, 0.0)
        torch.nn.init.normal_(self.alpha, mean=0.0, std=1)
        torch.nn.init.normal_(self.betta, mean=0.0, std=1)
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.output_gain:
            nn.init.ones_(self.gain)            

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        # self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
        # self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))
        with torch.no_grad():
            self.alpha.copy_(torch.from_numpy(alpha))
            self.betta.copy_(torch.from_numpy(betta))

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
        with torch.no_grad():
            self.test_forward = True
            sigmoid_func = torch.nn.Sigmoid()
            alpha_prob = sigmoid_func(self.alpha)
            betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
            prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
            # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
            # self.test_weight = torch.tensor(sampled, dtype=self.tensor_dtype, device=self.device)

            # self.num_of_options = num_of_options
            # self.test_weight_arr = []
            if tickets > 1:
                m = torch.distributions.Multinomial(tickets, prob_mat)
            else:
                m = torch.distributions.Categorical(prob_mat)
    #         for idx in range(0, self.num_of_options):
            sampled = m.sample()
            if tickets > 1:
                values = torch.argmax(sampled, dim=4) - 1
            else:
                values = sampled - 1
            # self.test_weight_arr.append(values)
            
#             self.test_weight = torch.tensor(values,dtype=self.tensor_dtype,device=self.device)
            test_weight = torch.tensor(values,dtype=self.tensor_dtype,device=self.device)
            self.test_weight = torch.nn.Parameter(test_weight)


    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            if self.output_gain:
                y = F.conv2d(input, self.test_weight, None, self.stride, self.padding, self.dilation, self.groups)
#                 return (y * self.gain[None, :, None, None]) + self.bias[None, :, None, None]
                abs_gain = torch.abs(self.gain)
                if self.bias is not None:
                    return (y * abs_gain[None, :, None, None]) + self.bias[None, :, None, None]
                else:
                    return (y * abs_gain[None, :, None, None])                    
            else:           
                # self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
                return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # if(self.in_channels == 128) and (self.out_channels == 128):
            #     print ("alpha: " + str(self.alpha))
            #     print ("betta: " + str(self.betta))
            #     print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
            #     print("betta isnan: " + str(torch.isnan(self.betta).any()))
            if torch.isnan(self.alpha).any():
                print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
            if torch.isnan(self.betta).any():
                print("betta isnan: " + str(torch.isnan(self.betta).any()))

            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            
            if self.output_gain:
                m = F.conv2d(input, mean, None, self.stride, self.padding, self.dilation, self.groups)
#                 m = m * self.gain[None, :, None, None] + self.bias[None, :, None, None]
                abs_gain = torch.abs(self.gain)
                if self.bias is not None:    
                    m = m * abs_gain[None, :, None, None] + self.bias[None, :, None, None]    
                else:
                    m = m * abs_gain[None, :, None, None]                    
            else:
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
            # z1 = torch.relu(z1) + self.eps  # TODO morning
#             z_bfr = z1
            if self.output_gain:                    
#                 gain_pow_2 = torch.pow(self.gain, 2)
#                 z1 = z1 * gain_pow_2[None, :, None, None]   
                abs_2 = self.gain * self.gain
                z1 = z1 * abs_2[None, :, None, None]                      
            z1 = z1 + self.eps ##TODO

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

            if torch.isnan(v).any():
                print("channels are: ", str(self.in_channels), str(self.out_channels))
#                 print("input: " + str(input))
                print("input isnan: " + str(torch.isnan(input).any()))
                print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
                print("betta isnan: " + str(torch.isnan(self.betta).any()))
#                 print_full_tensor(z1,"z1")
                print("z1 isnan: " + str(torch.isnan(z1).any()))
                print("z1 isinf: " + str(torch.isinf(z1).any()))
                print("z1 is negative: " + str((z1 < 0).any()))
                # print_neg_val(z_bfr, "z_bfr")
                # print_neg_val(z1, "z1")
                print("v isnan: " + str(torch.isnan(v).any()))
                exit(1)

            if self.output_sample:
                epsilon = torch.normal(0, 1, size=z1.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
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
        eps: int = 1e-05, # TODO today
    ):
        super(LRnetConv2d_ver2, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
        self.transposed = transposed
        self.output_sample = output_sample
        self.eps = eps
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
        torch.nn.init.normal_(self.alpha, mean=0.0, std=1)
        torch.nn.init.normal_(self.betta, mean=0.0, std=1)
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
        self.test_forward = False

    def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
        with torch.no_grad():
            self.test_forward = True
            sigmoid_func = torch.nn.Sigmoid()
            alpha_prob = sigmoid_func(self.alpha)
            betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
            prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)

            # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
            # self.test_weight = torch.tensor(sampled, dtype=self.tensor_dtype, device=self.device)

            # self.num_of_options = num_of_options
    #         self.test_weight_arr = []
            if tickets > 1:
                m = torch.distributions.Multinomial(tickets, prob_mat)
            else:
                m = torch.distributions.Categorical(prob_mat)
    #         for idx in range(0, self.num_of_options):
            sampled = m.sample()
            if tickets > 1:
                values = torch.argmax(sampled, dim=4) - 1
            else:
                values = sampled - 1
    #         self.test_weight_arr.append(values)
            self.test_weight = torch.tensor(values,dtype=self.tensor_dtype,device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            sign_input = torch.sign(input)
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
            # input_mean = 2 * (1 - torch.erf((-1) * m / v)) - 1
            cdf = 0.5 * (1 + torch.erf((-1) * m / (v * np.sqrt(2) + self.eps)))
            p = 1 - cdf
            input_mean = 2 * p - 1

            # print("input_mean: " + str(input_mean))

            m1 = F.conv2d(input_mean, mean, self.bias, self.stride, self.padding, self.dilation, self.groups) # TODO bias
            # print("m1: " + str(m1))
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean

            # sigma_square = mean_square - mean_pow2
#             e_h_2 = torch.ones(m.size(), dtype=self.tensor_dtype, device=self.device)
            e_h_2 = 1*p + 1*(1-p)

            # print("mean_square_tmp: " + str(mean_square_tmp))
            # print("mean_square: " + str(mean_square))
            # print("e_h_2: " + str(e_h_2))

            # if torch.cuda.is_available():
            #     torch.backends.cudnn.deterministic = True

            # z1 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)
            # z1 = m1 * m1
            z2 = F.conv2d(e_h_2, mean_square, None, self.stride, self.padding, self.dilation, self.groups)
            z3 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)

            # if torch.cuda.is_available():
            #     torch.backends.cudnn.deterministic = False

            # print("z2: " + str(z2))
            # print("z3: " + str(z3))

            # z = z1 + z2 - z3
            z = z2 - z3
            # z = torch.relu(z) + self.eps  # TODO morning
            z_bfr = z # TODO
            z = z + self.eps # TODO
            v1 = torch.sqrt(z)

            if torch.isnan(v1).any() or (z < 0).any():
                print("channels are: ", str(self.in_channels), str(self.out_channels))
                # print("alpha: \n" + str(self.alpha[31]))
                # print("betta: \n" + str(self.betta[31]))
                print("input_mean: \n" + str(input_mean[157]))
                print("m isnan: " + str(torch.isnan(m).any()))
                print("v isnan: " + str(torch.isnan(m).any()))
                print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
                print("betta isnan: " + str(torch.isnan(self.betta).any()))
                print("m1 isnan: " + str(torch.isnan(m1).any()))
                print("z is negative: " + str((z < 0).any()))
                print("z2 is negative: " + str((z2 < 0).any()))
                print("z3 is negative: " + str((z3 < 0).any()))
                print("z2 < z3: " + str((z2 < z3).any()))
                print("v1 isnan: " + str(torch.isnan(v1).any()))
                input_mean_2 = input_mean * input_mean
                print_fullllll_tensor(input_mean_2[157], "input_mean_2[157]")
                print("mean_pow2[31]: \n" +str(mean_pow2[31]))

                print_full_tensor(z2, "z2")
                print_full_tensor(z3, "z3")
                # print_neg_val(z_bfr, "z_bfr")
                print("e_h_2: \n" +str(e_h_2.size()))
                print("mean_square: \n" +str(mean_square.size()))
                print("input_mean^2.size: \n" +str(input_mean_2[157].size()))
                print("mean_pow2: \n" +str(mean_pow2.size()))
                exit(1)

            # print ("m: " + str(m))
            # print ("v: " + str(v))

            # print("z: " + str(z))
            # exit(1)

            # print("m1 isnan: " + str(torch.isnan(m1).any()))
            # print("v1 isnan: " + str(torch.isnan(v1).any()))
            # print("z is negative: " + str((z < 0).any()))

            if self.output_sample:
                epsilon = torch.normal(0, 1, size=z.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
                return m1 + epsilon * v1
            else:
                return m1, v1

# class LRBatchNorm2d(nn.BatchNorm2d):
#     def __init__(
#         self,
#         channels: int,
#         eps: int = 1e-05,
#         momentum: int = 0.9,
#         test_forward: bool = False
#     ):
#         super(LRBatchNorm2d, self).__init__()
#         self.channels, self.eps, self.test_forward, self.momentum = channels, eps, test_forward, momentum

class LRBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, test_forward = False):
        super(LRBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.test_forward = test_forward
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tensor_dtype = torch.float32
        self.use_batch_stats = True
        self.collect_stats = False

        # self.weight = torch.nn.Parameter(torch.ones([self.channels], dtype=torch.float32, device=self.device))
        # self.bias = torch.nn.Parameter(torch.zeros([self.channels], dtype=self.tensor_dtype, device=self.device))
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def use_batch_stats_switch(self, new_val) -> None:
        self.use_batch_stats = new_val

    def collect_stats_switch(self, new_val) -> None:
        self.collect_stats = new_val

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self) -> None:
        self.test_forward = True

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            # return input
#             if self.collect_stats:
#                 # print("branch 0")
#                 mean = input.mean([0, 2, 3])
#                 # use biased var in train
#                 var = input.var([0, 2, 3], unbiased=False)
#                 n = input.numel() / input.size(1)
#                 with torch.no_grad():
#                     self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
#                     # update running_var with unbiased var
#                     self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
#             else:
#                 # return self.bn(input)
#                 mean = self.running_mean
#                 var = self.running_var
            # TODO
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False)                
            output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
            if self.affine:
                output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            return output
        else:
            if self.training or self.use_batch_stats:
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

#                 weights_tmp = self.weight.repeat(m.size(0), 1)
#                 iweights = weights_tmp.view(m.size(0), m.size(1), 1, 1)
#                 bias_tmp = self.weight.repeat(m.size(0), 1)
#                 ibias = bias_tmp.view(m.size(0), m.size(1), 1, 1)

                # self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean

                # print("mean size: " + str(mean.size()))
                # print("mean_square size: " + str(mean_square.size()))
                # print("sigma_square size: " + str(sigma_square.size()))
                # print("iweights size: " + str(iweights.size()))
                # print("ibias size: " + str(ibias.size()))

                variance = sigma_square + mean_square - (mean * mean) + self.eps
                # variance = torch.relu(variance) + self.eps # TODO morning
                std = torch.sqrt(variance)

                norm_m = ((m - mean) / std)
                norm_v = (v / std)
                if self.affine:
                    norm_m = norm_m * self.weight[None, :, None, None] + self.bias[None, :, None, None]
                    norm_v = norm_v * self.weight[None, :, None, None]

                if torch.isnan(mean).any():
                    print("channels are: " + str(self.channels))
                    print("m isnan: " + str(torch.isnan(m).any()))
                    print("v isnan: " + str(torch.isnan(v).any()))
                    print("mean isnan: " + str(torch.isnan(mean).any()))
                    print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                    print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                    exit(1)

                if torch.isnan(variance).any():
                    print("channels are: " + str(self.channels))
                    print("m isnan: " + str(torch.isnan(m).any()))
                    print("v isnan: " + str(torch.isnan(v).any()))
                    print("variance isnan: " + str(torch.isnan(variance).any()))
                    print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                    print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                    exit(1)

                if torch.isnan(std).any():
                    print("channels are: " + str(self.channels))
                    print("variance: \n" + str(variance))
                    print("variance is negative: " + str((variance < 0).any()))
                    print("m isnan: " + str(torch.isnan(m).any()))
                    print("v isnan: " + str(torch.isnan(v).any()))
                    print("variance isnan: " + str(torch.isnan(variance).any()))
                    print("std isnan: " + str(torch.isnan(std).any()))
                    print("norm_m isnan: " + str(torch.isnan(norm_m).any()))
                    print("norm_v isnan: " + str(torch.isnan(norm_v).any()))
                    exit(1)
                return norm_m, norm_v
            else:
                print ("wrong branch")
                exit(1)
                # print("branch 0")
                mean = input.mean([0, 2, 3])
                # use biased var in train
                var = input.var([0, 2, 3], unbiased=False)
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

                input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
                if self.affine:
                    input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

                return input

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


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.use_batch_stats = False
        self.collect_stats = False
        self.use_test_stats = False

        self.test_running_mean = 0.0
        self.test_running_var = 0.0

    def update_use_batch_stats(self, new_val):
        self.use_batch_stats = new_val

    def update_collect_stats(self, new_val):
        self.collect_stats = new_val

    def update_use_test_stats(self, new_val):
        self.use_test_stats = new_val

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training or self.use_batch_stats:
            # print("branch 0")
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            # print("branch 1")
            mean = self.running_mean
            var = self.running_var

        if self.collect_stats:
            # print("branch 2")
            test_mean = input.mean([0, 2, 3])
            # use biased var in train
            test_var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.test_running_mean = self.momentum * test_mean\
                    + (1 - self.momentum) * self.test_running_mean
                # update running_var with unbiased var
                self.test_running_var = self.momentum * test_var * n / (n - 1)\
                    + (1 - self.momentum) * self.test_running_var

        if self.use_test_stats:
            # print("branch 4")
            mean = self.test_running_mean
            var = self.test_running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

    
class LRnet_sign_prob(nn.Module):

    def __init__(
        self,
        test_forward: bool = False,
        output_sample: bool = False,
        eps: int = 1e-05, # TODO today
    ):
        super(LRnet_sign_prob, self).__init__()
        self.test_forward = test_forward
        self.output_sample = output_sample
        self.eps = eps
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options, tickets=1) -> None:
        self.test_forward = True

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return torch.sign(input)
        else:
            m, v = input

            # mean of input
            # input_mean = 2 * (1 - torch.erf((-1) * m / v)) - 1
            cdf = 0.5 * (1 + torch.erf((-1) * m / (v * np.sqrt(2) + self.eps)))
            p = (1 - cdf)
            m1 = 2 * p - 1

            # sigma_square = mean_square - mean_pow2
#             e_h_2 = torch.ones(m.size(), dtype=self.tensor_dtype, device=self.device)
            e_h_2 = 1*p + 1*(1-p)
            
            z = e_h_2 - (m1*m1) + self.eps # TODO;
            v1 = torch.sqrt(z)

            if self.output_sample:
                epsilon = torch.normal(0, 1, size=z.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
                return m1 + epsilon * v1
            else:
                return m1, v1    

class LRnetLinear(nn.Module):
    def __init__(self, size_in, size_out, output_sample = True, test_forward = False, eps: int = 1e-05, sampled_input=False):
        super(LRnetLinear, self).__init__()        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32
   
        self.size_in, self.size_out = size_in, size_out
        self.weight = nn.Parameter(torch.ones(size_out, size_in, device=self.device))
        self.bias = nn.Parameter(torch.empty(size_out, device=self.device))
        self.output_sample = output_sample
        self.test_forward = test_forward
        self.eps = eps       
        self.sampled_input = sampled_input

        # initialize weight and biases
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # weight init
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options, tickets=1) -> None:
        self.test_forward = True
        
    def forward(self, input):
        if self.test_forward:
            w_times_x = torch.mm(input, self.weight.t())
            return torch.add(w_times_x, self.bias)  # w times x + b
        else:
            if self.sampled_input:
                w_times_x = torch.mm(input, self.weight.t())
                return torch.add(w_times_x, self.bias)  # w times x + b            
            else:
                m, v = input
                m_times_x = torch.mm(m, self.weight.t())
                m1 = torch.add(m_times_x, self.bias)  # m times x + b
                v1 = torch.sqrt(torch.mm(v*v, self.weight.t()*self.weight.t()) + self.eps)                                              
                
                if self.output_sample:
                    epsilon = torch.normal(0, 1, size=m1.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
    #                 epsilon = torch.normal(0, 1, size=m1.size(), requires_grad=False, device=self.device)                
                    return m1 + epsilon * v1
                else:
                    return m1, v1
                   


class LRnetConv2d_ver2X(nn.Module):

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
        sampled_input : bool = False,
        output_gain : bool = False, 
        eps: int = 1e-05, # TODO today
    ):
        super(LRnetConv2d_ver2X, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
        self.transposed = transposed
        self.output_sample = output_sample
        self.sampled_input = sampled_input
        self.output_gain = output_gain
        self.eps = eps
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

        if self.output_gain:
            self.gain = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))        

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
        torch.nn.init.normal_(self.alpha, mean=0.0, std=1)
        torch.nn.init.normal_(self.betta, mean=0.0, std=1)
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.output_gain:
            nn.init.ones_(self.gain)            

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
        with torch.no_grad():
            self.test_forward = True
            sigmoid_func = torch.nn.Sigmoid()
            alpha_prob = sigmoid_func(self.alpha)
            betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
            prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)

            if tickets > 1:
                m = torch.distributions.Multinomial(tickets, prob_mat)
            else:
                m = torch.distributions.Categorical(prob_mat)
            sampled = m.sample()
            if tickets > 1:
                values = torch.argmax(sampled, dim=4) - 1
            else:
                values = sampled - 1
            self.test_weight = torch.tensor(values,dtype=self.tensor_dtype,device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            if self.output_gain:
                y = F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)                
#                 return (y * self.gain[None, :, None, None]) + self.bias[None, :, None, None]
                abs_gain = torch.abs(self.gain)
                return (y * abs_gain[None, :, None, None]) + self.bias[None, :, None, None]    
            else:
                return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:            
            if self.sampled_input:
                prob_alpha = self.sigmoid(self.alpha)
                prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
                prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
                # E[X] calc
                mean_tmp = prob_mat * self.discrete_mat
                mean = torch.sum(mean_tmp, dim=4)

                if self.output_gain:
                    m = F.conv2d(input, mean, None, self.stride, self.padding, self.dilation, self.groups)
#                     m = m * self.gain[None, :, None, None] + self.bias[None, :, None, None]
                    abs_gain = torch.abs(self.gain)
                    m = m * abs_gain[None, :, None, None] + self.bias[None, :, None, None]
                else:                
                    m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    
                # E[x^2]
                mean_square_tmp = prob_mat * self.discrete_square_mat
                mean_square = torch.sum(mean_square_tmp, dim=4)
                # E[x] ^ 2
                mean_pow2 = mean * mean
                sigma_square = mean_square - mean_pow2
                z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)

                if self.output_gain:                    
#                     gain_pow_2 = torch.pow(self.gain, 2)
#                     z1 = z1 * gain_pow_2[None, :, None, None]
                    gain_2 = self.gain * self.gain
                    z1 = z1 * gain_2[None, :, None, None]
                
                z1 = z1 + self.eps ##TODO
                    
                v = torch.sqrt(z1)
                if self.output_sample:
                    epsilon = torch.normal(0, 1, size=z1.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
                    return (m + epsilon * v)
                else:
                    return m, v                
            else:
                m, v = input

                prob_alpha = self.sigmoid(self.alpha)
                prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
                prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
                # E[X] calc
                mean_tmp = prob_mat * self.discrete_mat
                mean = torch.sum(mean_tmp, dim=4)

                if self.output_gain:
                    m1 = F.conv2d(m, mean, None, self.stride, self.padding, self.dilation, self.groups)                    
#                     m1 = m1 * self.gain[None, :, None, None] + self.bias[None, :, None, None]
                    abs_gain = torch.abs(self.gain)
                    m1 = m1 * abs_gain[None, :, None, None] + self.bias[None, :, None, None]
                else:                
                    m1 = F.conv2d(m, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
                    
                # E[x^2]
                mean_square_tmp = prob_mat * self.discrete_square_mat
                mean_square = torch.sum(mean_square_tmp, dim=4)
                # E[x] ^ 2
                mean_pow2 = mean * mean

                z2 = F.conv2d(v, mean_square, None, self.stride, self.padding, self.dilation, self.groups)
                z3 = F.conv2d((m*m), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)

                z = z2 - z3
                z_bfr = z # TODO
                if self.output_gain:
#                     gain_pow_2 = torch.pow(self.gain, 2)
#                     z = z * gain_pow_2[None, :, None, None]
                    gain_2 = self.gain * self.gain
                    z = z * gain_2[None, :, None, None]                    
                z = z + self.eps # TODO
                v1 = torch.sqrt(z)                              

                if self.output_sample:
                    epsilon = torch.normal(0, 1, size=z.size(), dtype=self.tensor_dtype, requires_grad=False, device=self.device)
                    return m1 + epsilon * v1
                else:
                    return m1, v1   
            
class LRnet_sign_probX(nn.Module):

    def __init__(
        self,
        test_forward: bool = False,
        output_sample: bool = False,
        collect_stats: bool = False,
        bn_layer: bool = False,
        output_chan: int = 128,
        hard: bool = True,
        tau: int = 1.0,
        zero_act: int = 0,
        eps: int = 1e-10, # TODO today
    ):
        super(LRnet_sign_probX, self).__init__()
        self.test_forward = test_forward
        self.output_sample = output_sample
        self.eps = eps
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32
        
        self.collect_stats = collect_stats
        self.zero_act = zero_act

        if self.zero_act > 0:
            self.act_val = torch.ones(3, 1, dtype=torch.float32, requires_grad=False, device=self.device)
            self.act_val.copy_(torch.from_numpy(np.array([[-1],[0],[1]])))
        else:
            self.act_val = torch.ones(2, 1, dtype=torch.float32, requires_grad=False, device=self.device)
            self.act_val.copy_(torch.from_numpy(np.array([[-1],[1]])))

#         self.probability = torch.nn.Parameter(torch.empty([output_chan], dtype=self.tensor_dtype, device=self.device))
#         self.probability = torch.nn.Parameter(torch.empty([output_chan], dtype=torch.FloatTensor, device=self.device))
            
        self.tau = tau
        self.hard = hard
        
        self.bn_layer = bn_layer
        
        if self.bn_layer:      
            self.bn = LocalLRBatchNorm2d(output_chan, affine=True) # adding BN
#             self.bn = NewLocalLRBatchNorm2d(output_chan, affine=True) # adding BN
        
    def train_mode_switch(self) -> None:
        self.test_forward = False
        if self.bn_layer:              
            self.bn.test_forward = False  # adding BN

    def test_mode_switch(self, num_of_options, tickets=1) -> None:
        self.test_forward = True
        if self.bn_layer:                      
            self.bn.test_forward = True  # adding BN

    def forward(self, input: Tensor, writer=None, iteration_train=0, iterations_list=[], name='') -> Tensor:
        if self.test_forward:
            if self.bn_layer:                          
                return torch.sign(self.bn(input)) # adding BN
            else:
                if self.zero_act > 0:
                    tensor1 = (input <= self.zero_act) * (-1)
                    tensor2 = (input >= self.zero_act) * (1)
                    tensor1 = tensor1.type(input.type())
                    tensor2 = tensor2.type(input.type())
                    return (tensor1 + tensor2)
                else:
                    return torch.sign(input)        
        else:
            m, v = input
            
            if self.bn_layer:                              
                res = self.bn(input)  # adding BN
                m, v = res
                if writer is not None:
                    print("LRnet_sign_probX iteration_train:", iteration_train)   
                    print("bn input:")   
                    m_in, v_in = input
                    calc_batch_m_v(m_in, v_in, self.bn, 1, name+'_clt/'+name, iteration_train, iterations_list)
                    print("bn output:")   
                    calc_batch_m_v(m, v, self.bn, 1, name+'_clt/'+name, iteration_train, iterations_list)
#                     m_s, v_s = calc_m_v_sample (input, self.bn, 2000, name+'_clt/'+name, writer, iteration_train, iterations_list, rand_input=True)
                    m_a, v_a = calc_m_v_analyt (input, self.bn, 2000, name+'_clt/'+name, writer, iteration_train, iterations_list)
                    print(name + "_" + str(iteration_train) + ": m calc is: ", m[0][0][0][0])
                    print(name + "_" + str(iteration_train) + ": v calc is: ", v[0][0][0][0])    
                    m_s, v_s = m[0][0][0][0], v[0][0][0][0]
                    compare_m_v (m_a, v_a, m_s, v_s, "compare_m_v_"+name+"/iteraion", writer, iteration_train, iterations_list)                   
#                 print("m:", m)
#                 print("v:", v)

            if self.zero_act > 0:
                cdf_1 = 0.5 * (1 + torch.erf((self.zero_act - m) / (v * np.sqrt(2) + self.eps)))
                p_1 = (1 - cdf_1)
                cdf_m1 = 0.5 * (1 + torch.erf((-self.zero_act-m) / (v * np.sqrt(2) + self.eps)))
                p_m1 = cdf_m1
                m1 = 2 * p_1 - 1
                v1 = 1*p_1 + 1*(1-p_m1)
                p = p_1
            else:
                # mean of input
                cdf = 0.5 * (1 + torch.erf((-1) * m / (v * np.sqrt(2) + self.eps)))
                p = (1 - cdf)
                m1 = 2 * p - 1

    #             # sigma_square = mean_square - mean_pow2
    # #             e_h_2 = torch.ones(m.size(), dtype=self.tensor_dtype, device=self.device)
    #             e_h_2 = 1*p + 1*(1-p)

    #             z = e_h_2 - (m1*m1) + self.eps # TODO;
    #             v1 = torch.sqrt(z)

                v1 = 1*p + 1*(1-p)
    
            if self.output_sample:
                if self.zero_act > 0:
                    p0 = 1 - p_1 - p_m1
                    logit0 = torch.log(p_m1 + self.eps)
                    logit1 = torch.log(p0 + 1e-7)
                    logit2 = torch.log(p_1 + self.eps)
                    prob_mat = torch.stack((logit0, logit1, logit2), dim=4)
                    if torch.isnan(logit1).any() or ((p0 + 1e-7) < 0).any():
                        print("logit1 isnan: " + str(torch.isnan(logit1).any()))                    
                        print("p0 is negative: " + str((p0 < 0).any()))
                        print("p0 + 1e-7 is negative: " + str(((p0 + 1e-7) < 0).any()))
                        exit(1)
#                     print("prob_mat size", prob_mat.size())
# #                     print("logit1: " + str(logit1))                                        
# #                     print("logit1: " + str(p0))    
#                     print("p0: " + str(p0))    
#                     print("p0 is negative: " + str((p0 < 0).any()))
#                     print("p_1 + p_m1 > 1: " + str(((p_1 + p_m1) > 0).any()))
#                     print("p_1 + p_m1 max: " + str(torch.max(p_1 + p_m1)))
#                     print("1 - (p_1 + p_m1) min: " + str(torch.min(1 - (p_1 + p_m1))))
#                     print("p0 min: " + str(torch.min(p0)))
#                     print("p0 + 1e-5: " + str(torch.min(p0 + 1e-5)))
#                     print("logit0 isnan: " + str(torch.isnan(logit0).any()))                    
#                     print("logit1 isnan: " + str(torch.isnan(logit1).any()))                    
#                     print("logit2 isnan: " + str(torch.isnan(logit2).any()))     
#                     print_full_tensor((p_1 + p_m1),"(p_1 + p_m1)")
                else:
                    prob_mat = torch.stack((torch.log(1 - p + self.eps), torch.log(p + self.eps)), dim=4)
#                     print("self.probability size:",self.probability.size())
#                     print("self.probability type:",self.probability.type())
#                     print("p.mean([0,2,3]) size:",p.mean([0,2,3]).size())
#                     print("p.mean([0,2,3]) type:",p.mean([0,2,3]).type())
#                     self.probability = p.mean([0,2,3])
#                 print("prob_mat isnan: " + str(torch.isnan(prob_mat).any()))
#                 print("prob_mat isinf: " + str(torch.isinf(prob_mat).any()))
                z = F.gumbel_softmax(prob_mat, tau=self.tau, hard=self.hard, eps=1e-10, dim=-1)            
                z = torch.matmul(z, self.act_val)
                z = torch.squeeze(z)
#                 print("z", z)
#                 print("z isnan: " + str(torch.isnan(z).any()))
#                 print("z size", z.size())
#                 print("--------------------------------------")
#                 self.tau = 0.99*self.tau
#                 if torch.isnan(z).any():
#                     print("z isnan: " + str(torch.isnan(input).any()))
#                     print("z size is: ", str(z.size())) 
#                     exit(1)
                if self.collect_stats:
                    return z, p
                else:
                    return z
            elif self.collect_stats:
                return m1, v1, p
            else:            
                return m1, v1


# class LRnet_sign_probX(nn.Module):

#     def __init__(
#         self,
#         test_forward: bool = False,
#         output_sample: bool = False,
#         eps: int = 1e-05, # TODO today
#     ):
#         super(LRnet_sign_probX, self).__init__()
#         self.test_forward = test_forward
#         self.output_sample = output_sample
#         self.eps = eps
#         if torch.cuda.is_available():
#             self.device = 'cuda'
#         else:
#             self.device = 'cpu'
#         self.tensor_dtype = torch.float32
        
#         self.act_val = torch.ones(2, 1, dtype=torch.float32, requires_grad=False, device=self.device)
#         self.act_val.copy_(torch.from_numpy(np.array([[-1],[1]])))

#     def train_mode_switch(self) -> None:
#         self.test_forward = False

#     def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
#         self.test_forward = True

#     def forward(self, input: Tensor) -> Tensor:
#         if self.test_forward:
#             return torch.sign(input)
#         else:
#             m, v = input

#             # mean of input
#             cdf = 0.5 * (1 + torch.erf((-1) * m / (v * np.sqrt(2) + self.eps)))
#             p = (1 - cdf)
   
#             if self.output_sample:
#                 prob_mat = torch.stack(((1 - p), p), dim=4)
#                 z = F.gumbel_softmax(prob_mat, tau=0.1, hard=False, eps=1e-10, dim=-1)            
#                 z = torch.matmul(z, self.act_val)
#                 z = torch.squeeze(z)
# #                 m_dist = torch.distributions.Categorical(prob_mat)
# #                 sampled = m_dist.sample()
# # #                 print(sampled.size())
# # #                 print(sampled.type())
# # #                 print(v.type())
# # #                 sampled_output = 2*sampled - 1
# #                 sampled_output = sampled
# #                 sampled_output = 2*torch.tensor(sampled_output,dtype=self.tensor_dtype, device=self.device, requires_grad=False) - 1
# # #                 return sampled_output.type(torch.FloatTensor)
#                 return z
#             else:
#                 return m1, v1     

class LocalLRBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-10, momentum=0.1,
                 affine=True, track_running_stats=True, test_forward = False, act_norm=False, only_mean=False):
        super(LocalLRBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.test_forward = test_forward
        self.act_norm = act_norm
        self.only_mean = only_mean
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tensor_dtype = torch.float32
        
        if act_norm:
            self.weight = torch.nn.Parameter(torch.zeros([num_features], dtype=self.tensor_dtype, device=self.device))
            self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=self.tensor_dtype, device=self.device))            
            self.div = torch.nn.Parameter(torch.ones([num_features], dtype=self.tensor_dtype, device=self.device))
        else:
            self.weight = torch.nn.Parameter(torch.ones([num_features], dtype=self.tensor_dtype, device=self.device))
            self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=self.tensor_dtype, device=self.device))            

#         self.use_batch_stats = True
#         self.collect_stats = False

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         if self.affine:
#             nn.init.ones_(self.weight)
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)

#     def use_batch_stats_switch(self, new_val) -> None:
#         self.use_batch_stats = new_val

#     def collect_stats_switch(self, new_val) -> None:
#         self.collect_stats = new_val

    def train_mode_switch(self) -> None:
        self.test_forward = False

#     def test_mode_switch(self) -> None:
    def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
        self.test_forward = True

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            if self.only_mean:
                mean = input.mean([0, 2, 3])
                output = input - mean[None, :, None, None]                
                if self.affine:
                    abs_w = torch.abs(self.weight)
                    output = output * abs_w[None, :, None, None] + self.bias[None, :, None, None]  
                return output
            elif self.act_norm:
                output = (input - self.weight[None, :, None, None]) / (self.div[None, :, None, None] + self.eps)              
                output = output + self.bias[None, :, None, None]   
                return output                
            else:
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3], unbiased=False)                
    #             var = input.std([0, 2, 3])                
                output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))                
                if self.affine:
                    abs_w = torch.abs(self.weight)
                    output = output * abs_w[None, :, None, None] + self.bias[None, :, None, None]
                return output
        else:
#             if self.training: # TODO: add support in not training (but testing the continous case)
            if True:
        
                if self.act_norm:
                    m, v = input                
                    norm_m = ((m - self.weight[None, :, None, None])/(self.div[None, :, None, None] + self.eps)) + self.bias[None, :, None, None]          
                    abs_div = torch.abs(self.div)
                    norm_v = v   / (abs_div[None, :, None, None] + self.eps)       
                else:
                    m, v = input
                    mean = m.mean([0, 2, 3])
    #                 print("mean:", mean)
                    m_square = m * m 
                    mean_square = m_square.mean([0, 2, 3])                
    #                 print("mean_square:", mean_square)
                    v_square = v * v 
                    sigma_square = v_square.mean([0, 2, 3])                
    #                 print("sigma_square:", sigma_square)


                    variance = sigma_square + mean_square - (mean * mean) + self.eps
    #                 variance = mean_square + (mean * mean) + self.eps
                    std = torch.sqrt(variance)

    #                 print("variance:", variance)                
    #                 print("std:", std)                
    #                 print(m.size())
    #                 print(mean.size())
    #                 print(std.size())

    #                 norm_v = torch.sqrt((v*v) / std[None, :, None, None])

    #                 print("norm_m:", norm_m.mean())                
    #                 print("norm_v:", norm_v)                

    #                 norm_m = ((m - mean) / std)
    #                 norm_v = (v / std)

    #                 print("variance:", variance)                
    #                 print("std:", std)                
                    if self.only_mean:
                        norm_m = m - mean[None, :, None, None]            
                        norm_v = v         
                    else:
                        norm_m = (m - mean[None, :, None, None]) / std[None, :, None, None]            
                        norm_v = v / std[None, :, None, None]  

                    if self.affine:
    #                     norm_m = (m - mean[None, :, None, None]) / std[None, :, None, None]            
    #                     norm_m = norm_m * self.weight[None, :, None, None] + self.bias[None, :, None, None]                
    #                     w_2 = self.weight * self.weight
    #                     norm_v = torch.sqrt(((v*v) * w_2[None, :, None, None]) / variance[None, :, None, None])    

                        abs_w = torch.abs(self.weight)
                        norm_m = norm_m * abs_w[None, :, None, None] + self.bias[None, :, None, None]
                        norm_v = norm_v * abs_w[None, :, None, None] 
  
                    
                return norm_m, norm_v


class NewLocalLRBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-10, momentum=0.1,
                 affine=False, track_running_stats=True, test_forward = False):
        super(NewLocalLRBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.test_forward = test_forward
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.m_weight = torch.nn.Parameter(torch.empty([num_features], device=self.device))
        self.v_weight = torch.nn.Parameter(torch.empty([num_features], device=self.device))            
        self.first_iter = True

        self.tensor_dtype = torch.float32
#         self.use_batch_stats = True
#         self.collect_stats = False

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         if self.affine:
#             nn.init.ones_(self.weight)
#             if self.bias is not None:
#                 nn.init.zeros_(self.bias)

#     def use_batch_stats_switch(self, new_val) -> None:
#         self.use_batch_stats = new_val

#     def collect_stats_switch(self, new_val) -> None:
#         self.collect_stats = new_val

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self) -> None:
        self.test_forward = True

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
#             mean = input.mean([0, 2, 3])
#             var = input.var([0, 2, 3], unbiased=False)                
#             output = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#             if self.affine:
#                 output = output * self.weight[None, :, None, None] + self.bias[None, :, None, None]

            output = (input - self.m_weight[None, :, None, None]) / (self.v_weight[None, :, None, None] + self.eps)

            return output
        else:
#             if self.training: # add support in not training (but testing the continous case)
            if True:
        
                m, v = input

                if self.first_iter:
                    with torch.no_grad():
                        N = 1000
                        mean_vevtor = np.empty(self.num_features)
                        std_vevtor = np.empty(self.num_features)
                        for i in range(self.num_features):
                            Xm = m[:,i,:,:]
                            Xv = v[:,i,:,:]
                            m_samples = torch.unsqueeze(Xm,3)
                            v_samples = torch.unsqueeze(Xv,3)
                            m_samples = m_samples.expand(Xm.size(0), Xm.size(1), Xm.size(1), N)
                            v_samples = v_samples.expand(Xv.size(0), Xv.size(1), Xv.size(2), N)
                            with torch.no_grad():
                                epsilon = torch.normal(0.0, 1.0, size=m_samples.size(), device='cuda')
                                isamples = m_samples + epsilon * v_samples    
                                mean_s = isamples.mean([0, 1, 2, 3])
                                std_s = isamples.std([0, 1, 2, 3])            
                                mean_vevtor[i] = mean_s
                                std_vevtor[i] = std_s
                        self.m_weight.copy_(torch.from_numpy(mean_vevtor))
                        self.v_weight.copy_(torch.from_numpy(std_vevtor))
                        self.first_iter = False

                norm_m = (m - self.m_weight[None, :, None, None]) / self.v_weight[None, :, None, None]
                norm_v = v / self.v_weight[None, :, None, None]                
                                                        
#                 N = 2#100#0#0#0                
#                 with torch.no_grad():
#                     m_samples = m.expand(N, m.size(0), m.size(1), m.size(2), m.size(3))
#                     v_samples = v.expand(N, v.size(0), v.size(1), v.size(2), v.size(3))
# #                     print("m_samples size", m_samples.size())
#                     epsilon = torch.normal(0.0, 1.0, size=m_samples.size(), device=self.device)
#                     samples = m_samples + epsilon * v_samples
#                     mean_s = samples.mean([0, 1, 3, 4])
#                     std_s = samples.std([0, 1, 3, 4])                                
                    
# #                     epsilon = torch.normal(0.0, 1.0, size=m.size(), device=self.device)
# #                     y = m + epsilon * v
# #                     samples = y
# #                     for i in range(N):
# # #                         print("idx:",i)
# #                         epsilon = torch.normal(0.0, 1.0, size=m.size(), device=self.device)
# #                         y = m + epsilon * v
# #                         samples = torch.cat([samples,y])
# # #                         print ("samples size", samples.size())
# #                     mean_s = samples.mean([0, 2, 3])
# #                     std_s = samples.std([0, 2, 3])                                  
#                 norm_m = (m - mean_s[None, :, None, None]) / std_s[None, :, None, None]
#                 norm_v = v / std_s[None, :, None, None]                
                
#                 if self.affine:
#                     norm_m = norm_m * self.weight[None, :, None, None] + self.bias[None, :, None, None]
#                     norm_v = norm_v * self.weight[None, :, None, None]
                    
                return norm_m, norm_v
            

class LRnetAvgPool2d(nn.Module):

    def __init__(
        self,
        test_forward: bool = False
    ):
        super(LRnetAvgPool2d, self).__init__()
        self.test_forward = test_forward
        
    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options, tickets=1) -> None:
        self.test_forward = True
            
    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return F.avg_pool2d(input, 2)
        else:                             
            m, v = input
            m = F.avg_pool2d(m, 2)
            v = v*v
            v = F.avg_pool2d(v, 2) / 4
            v = torch.sqrt(v)            
            return m, v

        
class LRgain(nn.Module):

    def __init__(
        self,
        num_features: int = 64,        
        test_forward: bool = False
    ):
        super(LRgain, self).__init__()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tensor_dtype = torch.float32
        
        self.test_forward = test_forward
        self.weight = torch.nn.Parameter(torch.ones([num_features], dtype=self.tensor_dtype, device=self.device))
        self.bias = torch.nn.Parameter(torch.zeros([num_features], dtype=self.tensor_dtype, device=self.device))           
        
    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options, tickets=1) -> None:
        self.test_forward = True
            
    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            w2_sqrt = torch.sqrt(self.weight * self.weight)                       
            mean = input.mean([0, 2, 3])
            output = ((input - mean[None, :, None, None]) * w2_sqrt[None, :, None, None]) + self.bias[None, :, None, None]    
            return output            
        else:                             
            m, v = input
            w2_sqrt = torch.sqrt(self.weight * self.weight)                       
            m_mean = m.mean([0, 2, 3])
            norm_m = ((m - m_mean[None, :, None, None]) * w2_sqrt[None, :, None, None]) + self.bias[None, :, None, None]
            norm_v = v * w2_sqrt[None, :, None, None]
            return norm_m, norm_v
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            