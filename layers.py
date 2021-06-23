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
        self.tensoe_dtype = torch.float32

        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensoe_dtype, device=self.device))

        discrete_prob = np.array([-1.0, 0.0, 1.0])
        discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
        self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensoe_dtype, device=self.device)
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
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensoe_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensoe_dtype, device=self.device))

    def test_mode_switch(self) -> None:
        print ("test_mode_switch")
        self.test_forward = True
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
        prob_mat = prob_mat.detach().cpu().clone().numpy()

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
                            values_arr = np.random.default_rng().multinomial(10, theta)
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
            self.test_weight = torch.tensor(self.test_weight_arr[self.cntr],dtype=self.tensoe_dtype,device=self.device)
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            print ("elf.alpha: " + str(self.alpha))
            print ("elf.betta: " + str(self.betta))
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
            epsilon = torch.rand(z1.size(), requires_grad=False, dtype=self.tensoe_dtype, device=self.device)

            print ("m: " + str(m))
            print ("v: " + str(v))

            return m + epsilon * v


class MyNewConv2d(nn.Module):

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
        super(MyNewConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward = test_forward
        self.transposed = transposed
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # transposed = True
        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.tensoe_dtype = torch.float32

        self.alpha = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.betta = torch.nn.Parameter(torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensoe_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensoe_dtype, device=self.device))

        # discrete_prob = np.array([-1.0, 0.0, 1.0])
        # prob_mat = np.tile(discrete_prob, [D_0, D_1, D_2, D_3, 1])
        # square_prob_mat = prob_mat * prob_mat
        # self.discrete_mat = torch.tensor(prob_mat, requires_grad=False, dtype=torch.float32, device=self.device)
        # self.discrete_square_mat = torch.tensor(square_prob_mat, requires_grad=False, dtype=torch.float32, device=self.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # self.reset_train_parameters()
        init.constant_(self.alpha, -0.69314)
        init.constant_(self.betta, 0.0)
        # init.kaiming_uniform_(self.alpha, a=math.sqrt(5))
        # init.kaiming_uniform_(self.betta, a=math.sqrt(5))
        # init.uniform_(self.weight_theta, -1, 1)
        # init.constant_(self.weight_theta, 1)
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, alpha, betta) -> None:
        print ("Initialize Weights")
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.tensoe_dtype, device=self.device))
        self.betta = nn.Parameter(torch.tensor(betta, dtype=self.tensoe_dtype, device=self.device))

    def test_mode_switch(self) -> None:
        print ("test_mode_switch")
        self.test_forward = True;
        print("Initializing Test Weights: \n")
        sigmoid_func = torch.nn.Sigmoid()
        alpha_prob = sigmoid_func(self.alpha)
        betta_prob = sigmoid_func(self.betta)  * (1 - alpha_prob)
        prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)

        prob_mat = prob_mat.detach().cpu().clone().numpy()

        my_array = []
        for i, val_0 in enumerate(prob_mat):
            my_array_0 = []
            for j, val_1 in enumerate(val_0):
                my_array_1 = []
                for m, val_2 in enumerate(val_1):
                    my_array_2 = []
                    for n, val_3 in enumerate(val_2):
                        # theta = val_3
                        values_arr = np.random.default_rng().multinomial(10, val_3)
                        values = np.nanargmax(values_arr) - 1
                        # print ("val_3: " + str(val_3))
                        # print ("values: " + str(values))
                        my_array_2.append(values)
                    my_array_1.append(my_array_2)
                my_array_0.append(my_array_1)
            my_array.append(my_array_0)
        self.test_weight = torch.tensor(my_array, dtype=self.tensoe_dtype, device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            # TODO: self.discrete_mat = self.discrete_mat.to(prob_mat.get_device())

            discrete_prob = np.array([-1.0, 0.0, 1.0])
            discrete_prob = np.tile(discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
            # discrete_mat = torch.tensor(discrete_prob, requires_grad=False, dtype=torch.float32, device='cuda')
            discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensoe_dtype, device='cuda')

            mean_tmp = prob_mat * discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # E[x^2]
            # TODO: self.discrete_square_mat = self.discrete_square_mat.to(prob_mat.get_device())
            square_discrete_prob = np.array([1.0, 0.0, 1.0])
            square_discrete_prob = np.tile(square_discrete_prob, [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
            # discrete_square_mat = torch.tensor(square_discrete_prob, requires_grad=False, dtype=torch.float32, device='cuda')
            discrete_square_mat = torch.as_tensor(square_discrete_prob, dtype=self.tensoe_dtype, device='cuda')

            mean_square_tmp = prob_mat * discrete_square_mat
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

            epsilon = torch.rand(z1.size(), requires_grad=False, dtype=self.tensoe_dtype, device='cuda')

            # epsilon = torch.rand(z1.size())
            # if torch.cuda.is_available():
            #     epsilon = epsilon.to(device='cuda')
            #     # epsilon = epsilon.to(z1.get_device())
            return m + epsilon * v
