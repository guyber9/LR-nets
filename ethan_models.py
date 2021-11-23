from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import torch.nn as nn
import time
import ethan_layers as lrnet_nn
from utils import print_full_tensor, assertnan, collect_hist, collect_m_v, take_silce, output_hist, layer_hist, id_generator, calc_avg_entropy, calc_avg_entropy3, get_p, get_x, compare_m_v, calc_m_v_sample, calc_m_v_analyt
import numpy as np
from utils import test


class LRNet_CIFAR10_act(nn.Module):

    def __init__(self, writer=None, args=None):
        super(LRNet_CIFAR10_act, self).__init__()

        self.args = args
        self.sampled_last_layer = False # always False here3
        
        self.wide = 1.00        
        self.bn_s = False
        self.gumbel = True        
        self.gumble_last_layer = False # True if not self.sampled_last_layer else False
        self.bn_last_layer = True or self.bn_s # True if not self.sampled_last_layer else False
        self.gain = True 
        
        self.dropout = False
        self.tau = 1.0
        
        self.l6 = 512
        
        self.pool = False
        self.stride = 1 if self.pool else 2
        if self.pool:
            self.pool2 = lrnet_nn.LRnetAvgPool2d()
            self.pool4 = lrnet_nn.LRnetAvgPool2d()
            if self.gumble_last_layer:
                self.pool6 = nn.AvgPool2d(2, stride=2)                
            else:
                self.pool6 = lrnet_nn.LRnetAvgPool2d()
                
        if self.wide > 1.0:
            self.flat_chan = int(8192 * self.wide)
        else:
            self.flat_chan = 8192 # 32768 # 16384 # 8192 # 4096             
                
        self.fc1_output_chan = 10

        self.conv1 = lrnet_nn.LRnetConv2d(3, int(128*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)

        if self.gumbel:
            self.conv2 = lrnet_nn.LRnetConv2d(int(128*self.wide), int(128*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv3 = lrnet_nn.LRnetConv2d(int(128*self.wide), int(256*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv4 = lrnet_nn.LRnetConv2d(int(256*self.wide), int(256*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)
            self.conv5 = lrnet_nn.LRnetConv2d(int(256*self.wide), int(self.l6*self.wide), 3, stride=1, padding=1, output_gain=self.gain, output_sample=False)
            self.conv6 = lrnet_nn.LRnetConv2d(int(self.l6*self.wide), int(self.l6*self.wide), 3, stride=self.stride, padding=1, output_gain=self.gain, output_sample=False)          
            
        self.sign_prob1 = lrnet_nn.LRnet_sign_probX(output_chan=int(128*self.wide), tau=self.tau, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob2 = lrnet_nn.LRnet_sign_probX(output_chan=int(128*self.wide), tau=self.tau, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob3 = lrnet_nn.LRnet_sign_probX(output_chan=int(256*self.wide), tau=self.tau, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob4 = lrnet_nn.LRnet_sign_probX(output_chan=int(256*self.wide), tau=self.tau, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        self.sign_prob5 = lrnet_nn.LRnet_sign_probX(output_chan=int(self.l6*self.wide), tau=self.tau, output_sample=self.gumbel, collect_stats=True, bn_layer = self.bn_s)
        
        self.sign_prob6 = lrnet_nn.LRnet_sign_probX(output_chan=int(self.l6*self.wide), tau=self.tau, output_sample=self.gumble_last_layer, collect_stats=True, bn_layer = self.bn_last_layer)
 
        if self.gumble_last_layer:
            self.fc1 = nn.Linear(self.flat_chan, self.fc1_output_chan)
        else:
            self.fc1 = lrnet_nn.LRnetLinear(self.flat_chan, self.fc1_output_chan)

#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.5)
    
        if writer is not None:
            self.writer = writer
            self.inner_iteration_test = 0
            self.inner_iteration_train = 0
        else:
            self.writer = None
        self.tensorboard_train = False
        self.tensorboard_test = False    
        self.iteration_train = 0      
        self.iteration_test = 0
        self.net_id = id_generator(16)
        self.sigmoid_func = torch.nn.Sigmoid()   
#         self.iterations_list = [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450]
        self.iterations_list = [0, 10, 50, 100, 200, 250, 350, args.epochs-1]
#         self.iterations_list1 = [0, 10, 50, 100, 200, 250, 270, 290, 310, 330, 350, args.epochs-1]
        self.iterations_list1 = self.iterations_list
        
    def forward(self, x):
        cin = x
        x = self.conv1(x)
        
        x = self.sign_prob1(x)
        if self.tensorboard_train and self.sign_prob1.collect_stats and not self.sign_prob1.test_forward and (self.writer is not None):
            p = get_p(x, self.sign_prob1.output_sample)
            p = p + 1e-10
            calc_avg_entropy (p, "sign_entropy", "/sign1_entropy", self.iteration_train, self.writer)
            alpha_p = self.sigmoid_func(self.conv1.alpha) + 1e-10            
            betta_p = self.sigmoid_func(self.conv1.betta) * (1 - alpha_p) + 1e-10        
            calc_avg_entropy3 (alpha_p, betta_p, "weights_entropy", "/weights_1", self.inner_iteration_train, self.writer)           
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

        x = self.fc1(x)  # 8192 -> 10 

        output = x

        return output

#         if self.sign_prob6.test_forward:
#             return output
#         else:
#             return output, p       
        
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
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()    
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.gumble_last_layer:
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
        if not self.gumble_last_layer:
            self.fc1.train_mode_switch()        
        if self.pool:
            self.pool2.train_mode_switch()
            self.pool4.train_mode_switch()
            if not self.gumble_last_layer:
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
        if not self.gumble_last_layer:
            self.fc1.test_mode_switch(1, 1)     
        if self.pool:
            self.pool2.test_mode_switch(1, 1)
            self.pool4.test_mode_switch(1, 1)
            if not self.gumble_last_layer:
                self.pool6.test_mode_switch(1, 1)  



###########################
## Full Precision Models ##
###########################

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

