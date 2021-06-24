from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

def train(args, model, device, train_loader, optimizer, epoch, f=None):
    model.train()
    weight_decay = 10**((-1)*args.wd) # 1e-4
    probability_decay = 10**((-1)*args.pd) # 1e-11
    print("weight_decay: " + str(weight_decay))
    print("probability_decay: " + str(probability_decay))
    # torch.backends.cudnn.benchmark = True
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # if args.parallel_gpu == 0:
        #     parallel_net = model
        # elif args.parallel_gpu == 1:
        #     parallel_net = nn.DataParallel(model, device_ids=[0])
        # elif args.parallel_gpu == 2:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1])
        # elif args.parallel_gpu == 3:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1, 2])
        # elif args.parallel_gpu == 4:
        #     parallel_net = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        # elif args.parallel_gpu == 5:
        #     dist.init_process_group(backend='nccl', rank=2, world_size=3, init_method = None, store = None)
        #     parallel_net = DDP(model, device_ids=[0, 1, 2], output_device=[0])
        # parallel_net = model

        optimizer.zero_grad()
        # output = parallel_net(data)
        output = model(data)

        if args.cifar10:
            if args.full_prec:
                loss = F.cross_entropy(output, target)
                ce_loss = loss
            else:
                ce_loss = F.cross_entropy(output, target)
                loss = ce_loss + probability_decay * (torch.norm(model.conv1.alpha, 2) + torch.norm(model.conv1.betta, 2)
                                                 + torch.norm(model.conv2.alpha, 2) + torch.norm(model.conv2.betta, 2)
                                                 + torch.norm(model.conv3.alpha, 2) + torch.norm(model.conv3.betta, 2)
                                                 + torch.norm(model.conv4.alpha, 2) + torch.norm(model.conv4.betta, 2)
                                                 + torch.norm(model.conv5.alpha, 2) + torch.norm(model.conv5.betta, 2)
                                                 + torch.norm(model.conv6.alpha, 2) + torch.norm(model.conv6.betta, 2)) \
                       + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
                        # + weight_decay * (torch.norm(model.fc1.bias, 2) + (torch.norm(model.fc2.bias, 2)))
        # + weight_decay * (torch.norm(model.conv1.bias, 2) + torch.norm(model.conv2.bias, 2) \
        #                   + torch.norm(model.conv3.bias, 2) + torch.norm(model.conv4.bias, 2) \
        #                   + torch.norm(model.conv5.bias, 2) + torch.norm(model.conv6.bias, 2)) \
        else:
            if args.full_prec:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.cross_entropy(output, target) + probability_decay * (torch.norm(model.conv1.alpha, 2)
                                                               + torch.norm(model.conv1.betta, 2)
                                                               + torch.norm(model.conv2.alpha, 2)
                                                               + torch.norm(model.conv2.betta, 2)) + weight_decay * (torch.norm(model.fc1.weight, 2) + (torch.norm(model.fc2.weight, 2)))
        # optimizer.zero_grad()

        if args.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), ce_loss.item(), loss.item()))
        if f is not None:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item(), loss.item()), file = f)

            if args.dry_run:
                break

def test(model, device, test_loader, test_mode, f=None):
    if test_mode:
        tstring = 'Test'
    else:
        tstring = 'Train'
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n' + str(tstring) +' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if f is not None:
        print('\n' + str(tstring) + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), file = f)
    return (100. * correct / len(test_loader.dataset))


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
        print("alpha: " + str(e_alpha))
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
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 3
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x) # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2) # 128 x 16 x 16
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x) # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2) # 256 x 8 x 8
        x = F.relu(x)
        # x = self.dropout4(x)

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
        # self.conv1 = mySigmConv2d(3, 128, 3, 1, padding=1)
        # self.conv2 = mySigmConv2d(128, 128, 3, 1, padding=1)
        # self.conv3 = mySigmConv2d(128, 256, 3, 1, padding=1)
        # self.conv4 = mySigmConv2d(256, 256, 3, 1, padding=1)
        # self.conv5 = mySigmConv2d(256, 512, 3, 1, padding=1)
        # self.conv6 = mySigmConv2d(512, 512, 3, 1, padding=1)
        self.conv1 = MyNewConv2d(3, 128, 3, 1, padding=1)
        self.conv2 = MyNewConv2d(128, 128, 3, 1, padding=1)
        self.conv3 = MyNewConv2d(128, 256, 3, 1, padding=1)
        self.conv4 = MyNewConv2d(256, 256, 3, 1, padding=1)
        self.conv5 = MyNewConv2d(256, 512, 3, 1, padding=1)
        self.conv6 = MyNewConv2d(512, 512, 3, 1, padding=1)
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
        # self.dropout3 = nn.Dropout(0.5)
        # self.dropout4 = nn.Dropout(0.5)
        # self.dropout5 = nn.Dropout(0.2)
        # self.dropout6 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        # utils.print_full_tensor(x, "x1")
        x = self.bn1(x)
        # utils.print_full_tensor(x, "x_bn1")
        x = F.relu(x)
        # x = self.dropout6(x)
        x = self.conv2(x)  # 128 x 32 x 32
        # utils.print_full_tensor(x, "x2")
        x = self.bn2(x)
        # utils.print_full_tensor(x, "x_bn2")
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        # x = self.dropout3(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.dropout5(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        # x = self.dropout4(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
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
        # print("output: " + str(output))
        return output

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='M',
                        help='Step size for scheduler (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--full-prec', action='store_true', default=False,
                        help='For Training Full Precision Model')
    parser.add_argument('--load-pre-trained', action='store_true', default=False,
                        help='For Loading Params from Trained Full Precision Model')
    parser.add_argument('--debug-mode', action='store_true', default=False, help='For Debug Mode')
    parser.add_argument('--cifar10', action='store_true', default=True, help='cifar10 flag')
    parser.add_argument('--resume', action='store_true', default=False, help='resume model flag')
    parser.add_argument('--parallel-gpu', type=int, default=1, metavar='N',
                        help='parallel-gpu (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num_workers (default: 4)')
    parser.add_argument('--save', action='store', default='saved_models/cifar10',
                        help='name of saved model')
    parser.add_argument('--num', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--wd', type=int, default=4, metavar='N',
                        help='wd is 10**((-1)*wd)')
    parser.add_argument('--pd', type=int, default=11, metavar='N',
                        help='pd is 10**((-1)*pd)')

    parser.add_argument('--bias', action='store_true', default=False, help='initial bias')
    parser.add_argument('--sgd', action='store_true', default=False, help='run with sgd')
    parser.add_argument('--sched', action='store_true', default=False, help='another sched')
    parser.add_argument('--fc', action='store_true', default=False, help='initial fc')
    parser.add_argument('--bn', action='store_true', default=False, help='initial batch norm')
    parser.add_argument('--norm', action='store_true', default=False, help='norm init layer')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    print('Reading Database..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset1 = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)

    dataset2 = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.full_prec:
        print ("Training FPNet_CIFAR10")
        model = FPNet_CIFAR10().to(device)
        if args.sgd:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=1e-2)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        print ("Training LRNet")
        model = LRNet_CIFAR10().to(device)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr,
        #                       momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        if args.sgd:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # optimizer = optim.SGD(model.parameters(), lr=args.lr,
        #                       momentum=0.9, weight_decay=1e-2)
        # optimizer = optim.Adam([
        #     {'params': model.conv1.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv2.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv3.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv4.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv5.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.conv6.parameters(), 'weight_decay': 1e-11},
        #     {'params': model.fc1.parameters(), 'weight_decay': 1e-4},
        #     {'params': model.fc2.parameters(), 'weight_decay': 1e-4}
        # ], lr=args.lr)

        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.load_pre_trained and not args.resume:
        print ("Loading Model")
        test_model = FPNet_CIFAR10().to(device)
        if use_cuda:
            test_model.load_state_dict(torch.load('saved_models/cifar10_fp.pt'))
        else:
            test_model.load_state_dict(torch.load('saved_models/cifar10_full_prec_no_cuda.pt'))
        # test_model.eval()
        # state_dict = torch.load('saved_models/cifar10_full_prec.pt')

        alpha1, betta1 = find_sigm_weights(test_model.conv1.weight, False)
        alpha2, betta2 = find_sigm_weights(test_model.conv2.weight, False)
        alpha3, betta3 = find_sigm_weights(test_model.conv3.weight, False)
        alpha4, betta4 = find_sigm_weights(test_model.conv4.weight, False)
        alpha5, betta5 = find_sigm_weights(test_model.conv5.weight, False)
        alpha6, betta6 = find_sigm_weights(test_model.conv6.weight, False)

        model.conv1.initialize_weights(alpha1, betta1)
        model.conv2.initialize_weights(alpha2, betta2)
        model.conv3.initialize_weights(alpha3, betta3)
        model.conv4.initialize_weights(alpha4, betta4)
        model.conv5.initialize_weights(alpha5, betta5)
        model.conv6.initialize_weights(alpha6, betta6)

        def normalize_layer(w):
            if args.norm:
                return nn.Parameter(w / torch.std(w))
            else:
                return w

        if args.bias:
            model.conv1.bias = normalize_layer(test_model.conv1.bias)
            model.conv2.bias = normalize_layer(test_model.conv2.bias)
            model.conv3.bias = normalize_layer(test_model.conv3.bias)
            model.conv4.bias = normalize_layer(test_model.conv4.bias)
            model.conv5.bias = normalize_layer(test_model.conv5.bias)
            model.conv6.bias = normalize_layer(test_model.conv6.bias)

        if args.fc:
            model.fc1.weight = test_model.fc1.weight
            model.fc1.bias = test_model.fc1.bias
            model.fc2.weight = test_model.fc2.weight
            model.fc2.bias = test_model.fc2.bias

        if args.bn:
            model.bn1.bias = test_model.bn1.bias
            model.bn1.weight = test_model.bn1.weight
            # model.bn1.running_mean = test_model.bn1.running_mean
            # model.bn1.running_var = test_model.bn1.running_var

            model.bn2.bias = test_model.bn2.bias
            model.bn2.weight = test_model.bn2.weight
            # model.bn2.running_mean = test_model.bn2.running_mean
            # model.bn2.running_var = test_model.bn2.running_var

            model.bn3.bias = test_model.bn3.bias
            model.bn3.weight = test_model.bn3.weight
            # model.bn3.running_mean = test_model.bn3.running_mean
            # model.bn3.running_var = test_model.bn3.running_var

            model.bn4.bias = test_model.bn4.bias
            model.bn4.weight = test_model.bn4.weight
            # model.bn4.running_mean = test_model.bn4.running_mean
            # model.bn4.running_var = test_model.bn4.running_var

            model.bn5.bias = test_model.bn5.bias
            model.bn5.weight = test_model.bn5.weight
            # model.bn5.running_mean = test_model.bn5.running_mean
            # model.bn5.running_var = test_model.bn5.running_var

            model.bn6.bias = test_model.bn6.bias
            model.bn6.weight = test_model.bn6.weight
            # model.bn6.running_mean = test_model.bn6.running_mean
            # model.bn6.running_var = test_model.bn6.running_var

    if args.resume and not args.load_pre_trained:
        print("Resume Model: LRNet  (if args.resume and not args.load_pre_trained)")
        model.load_state_dict(torch.load('saved_model/best_cifar10_cnn.pt'))

    if args.load_pre_trained and args.resume:
        print("if args.load_pre_trained and args.resume")
        test_model = LRNet_CIFAR10().to(device)
        test_model.load_state_dict(torch.load('saved_model/best_cifar10_cnn.pt'))
        model.conv1.alpha = test_model.conv1.alpha
        model.conv1.betta = test_model.conv1.betta
        model.conv1.bias = test_model.conv1.bias

        model.conv2.alpha = test_model.conv2.alpha
        model.conv2.betta = test_model.conv2.betta
        model.conv2.bias = test_model.conv2.bias

        model.conv3.alpha = test_model.conv3.alpha
        model.conv3.betta = test_model.conv3.betta
        model.conv3.bias = test_model.conv3.bias

        model.conv4.alpha = test_model.conv4.alpha
        model.conv4.betta = test_model.conv4.betta
        model.conv4.bias = test_model.conv4.bias

        model.conv5.alpha = test_model.conv5.alpha
        model.conv5.betta = test_model.conv5.betta
        model.conv5.bias = test_model.conv5.bias

        model.conv6.alpha = test_model.conv6.alpha
        model.conv6.betta = test_model.conv6.betta
        model.conv6.bias = test_model.conv6.bias

    print ("###################################")
    print ("training..")
    print ("num of epochs: " + str(args.epochs))
    print ("###################################")
    if args.full_prec:
        if args.sched:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        if args.sched:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    file_name = str(args.save) + ".log"
    f = open(file_name, "w")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # with torch.cuda.amp.autocast():
        train(args, model, device, train_loader, optimizer, epoch, f)
        print('{} seconds'.format(time.time() - t0))
        test(model, device, test_loader, True, f)
        if ((epoch % 30) == 0) or (epoch == args.epochs):
            print("Accuracy on train data:")
            # torch.save(model.state_dict(), "saved_models/cifar10_interim_model.pt")
            test(model, device, train_loader, False, f)
        scheduler.step()

    f.close()

    if args.full_prec:
        if use_cuda:
            torch.save(model.state_dict(), str(args.save) + "_full_prec.pt")
        else:
            torch.save(model.state_dict(), str(args.save) + "_full_prec_no_cuda.pt")
    else:
        torch.save(model.state_dict(), str(args.save) + "_cnn.pt")

if __name__ == '__main__':
    main()
