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
from utils import print_full_tensor

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
    ):
        super(LRnetConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
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

        # ################################################3
        # print_full_tensor(prob_mat, "prob_mat")
        # for idx in range(0, self.num_of_options):
        #     print_full_tensor(self.test_weight_arr[idx], "self.test_weight_arr__" + str(idx))
        # dif = self.test_weight_arr[1] - self.test_weight_arr[0]
        # print_full_tensor(dif, "dif")
        # histc = torch.histc(prob_mat, bins=10, min=0, max=1)
        # print(histc)
        # exit(1)
        # ################################################3

    # it was thst way
        # self.num_of_options = num_of_options
        # self.test_weight_arr = []
        # for idx in range(0, self.num_of_options):
        #     sampled = torch.distributions.Categorical(prob_mat).sample() - 1
        #     self.test_weight_arr.append(sampled)

        # self.test_weight_arr = []
        # for idx in range(0, self.num_of_options):
        #     self.test_weight_arr.append([])
        # for i, val_0 in enumerate(prob_mat):
        #     my_array_0 = []
        #     for idx in range(0, self.num_of_options):
        #         my_array_0.append([])
        #     for j, val_1 in enumerate(val_0):
        #         my_array_1 = []
        #         for idx in range(0, self.num_of_options):
        #             my_array_1.append([])
        #         for m, val_2 in enumerate(val_1):
        #             my_array_2 = []
        #             for idx in range(0, self.num_of_options):
        #                 my_array_2.append([])
        #             for n, theta in enumerate(val_2):
        #                 for idx in range(0, self.num_of_options):
        #                     values_arr = np.random.default_rng().multinomial(tickets, theta)
        #                     values = np.nanargmax(values_arr) - 1
        #                     my_array_2[idx].append(values)
        #             for idx in range(0, self.num_of_options):
        #                 my_array_1[idx].append(my_array_2[idx])
        #         for idx in range(0, self.num_of_options):
        #             my_array_0[idx].append(my_array_1[idx])
        #     for idx in range(0, self.num_of_options):
        #         self.test_weight_arr[idx].append(my_array_0[idx])

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            # print("Initializing Test Weights: \n")
            # sigmoid_func = torch.nn.Sigmoid()
            # alpha_prob = sigmoid_func(self.alpha)
            # betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
            # prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
            # sampled = torch.distributions.Categorical(prob_mat).sample() - 1
            # self.test_weight = torch.tensor(sampled,dtype=self.tensor_dtype,device=self.device)
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # print ("alpha: " + str(self.alpha))
            # print ("betta: " + str(self.betta))
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
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False
            v = torch.sqrt(z1)
            epsilon = torch.rand(z1.size(), requires_grad=False, dtype=self.tensor_dtype, device=self.device)

            # print ("m: " + str(m))
            # print ("v: " + str(v))

            return m + epsilon * v


class LRnetConv2d_not_sample(nn.Module):

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
    ):
        super(LRnetConv2d_not_sample, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
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
        # init.constant_(self.alpha, -0.69314)
        # init.constant_(self.betta, 0.0)
        # init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        # init.kaiming_uniform_(self.betta, a=math.sqrt(5))
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

    def test_mode_switch(self, num_of_options, tickets=10) -> None:
        print ("test_mode_switch")
        self.test_forward = True
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
        prob_mat = prob_mat.detach().cpu().clone().numpy()
        self.num_of_options = num_of_options
        self.test_weight_arr = []
        for idx in range(0, self.num_of_options):
            self.test_weight_arr.append([])
        for i, val_0 in enumerate(prob_mat):
            my_array_0 = []
            for idx in range(0, self.num_of_options):
                my_array_0.append([])
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for idx in range(0, self.num_of_options):
                    my_array_1.append([])
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for idx in range(0, self.num_of_options):
                        my_array_2.append([])
                    for n, theta in enumerate(val_2):
                        for idx in range(0, self.num_of_options):
                            values_arr = np.random.default_rng().multinomial(tickets, theta)
                            values = np.nanargmax(values_arr) - 1
                            my_array_2[idx].append(values)
                    for idx in range(0, self.num_of_options):
                        my_array_1[idx].append(my_array_2[idx])
                for idx in range(0, self.num_of_options):
                    my_array_0[idx].append(my_array_1[idx])
            for idx in range(0, self.num_of_options):
                self.test_weight_arr[idx].append(my_array_0[idx])

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
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
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False
            v = torch.sqrt(z1)

            # print ("m: " + str(m))
            # print ("v: " + str(v))

            return m, v

class NewLRnetConv2d(nn.Module):

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
        super(NewLRnetConv2d, self).__init__()
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

    def test_mode_switch(self, num_of_options, tickets=10) -> None:
        print ("test_mode_switch")
        self.test_forward = True
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
        prob_mat = prob_mat.detach().cpu().clone().numpy()
        self.num_of_options = num_of_options
        self.test_weight_arr = []
        for idx in range(0, self.num_of_options):
            self.test_weight_arr.append([])
        for i, val_0 in enumerate(prob_mat):
            my_array_0 = []
            for idx in range(0, self.num_of_options):
                my_array_0.append([])
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for idx in range(0, self.num_of_options):
                    my_array_1.append([])
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for idx in range(0, self.num_of_options):
                        my_array_2.append([])
                    for n, theta in enumerate(val_2):
                        for idx in range(0, self.num_of_options):
                            values_arr = np.random.default_rng().multinomial(tickets, theta)
                            values = np.nanargmax(values_arr) - 1
                            my_array_2[idx].append(values)
                    for idx in range(0, self.num_of_options):
                        my_array_1[idx].append(my_array_2[idx])
                for idx in range(0, self.num_of_options):
                    my_array_0[idx].append(my_array_1[idx])
            for idx in range(0, self.num_of_options):
                self.test_weight_arr[idx].append(my_array_0[idx])

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensor_dtype,device=self.device)
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            m, v = input
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
            # print_full_tensor(discrete_input, "")

            m1 = F.conv2d(input_mean, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean

            # sigma_square = mean_square - mean_pow2
            e_h_2 = torch.ones(input_mean.size(), dtype=self.tensor_dtype, device=self.device)

            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True

            # z1 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)
            # z1 = m1 * m1
            z2 = F.conv2d(e_h_2, mean_square, None, self.stride, self.padding, self.dilation, self.groups)
            z3 = F.conv2d((input_mean*input_mean), mean_pow2, None, self.stride, self.padding, self.dilation, self.groups)

            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = False

            # z = z1 + z2 - z3
            z = z2 - z3
            v1 = torch.sqrt(z)

            # print ("m: " + str(m))
            # print ("v: " + str(v))

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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            init.nn.zeros_(self.bias)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.weight = nn.Parameter(torch.tensor(alpha, dtype=self.tensor_dtype, device=self.device))
        self.bias = nn.Parameter(torch.tensor(betta, dtype=self.tensor_dtype, device=self.device))

    def test_mode_switch(self, num_of_options, tickets=10) -> None:
        return

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return
        else:
            m, v = input

            mean = torch.mean(m)
            mean_square = torch.mean(m * m)
            sigma_square = torch.mean(v * v)
            variance = sigma_square + mean_square - (mean * mean) + self.eps
            std = torch.sqrt(variance)

            norm_m = (m - mean) / std
            norm_v = v / variance

            return norm_m, norm_v
