# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-01-06 18:19:56
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-02-18 17:05:55
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time 
from user_profile_constant import *
from user_profile_common_function import *

tempEnv = gym.make('UserProfile-v0')
tempEnv = tempEnv.unwrapped
ENV_A_SHAPE = 0 if isinstance(tempEnv.action_space.sample(), int) else tempEnv.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_NOTES)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(N_NOTES, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        # print(x.shape)
        # print(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class FM_Layer(nn.Module):
    def __init__(self,):
        super(FM_Layer, self).__init__()
        self.n = Vector_N
        self.k = Embedding_K
        self.linear = nn.Linear(self.n, 1)   # 前两项线性层
        self.V = nn.Parameter(torch.randn(self.n, self.k))   # 交互矩阵
    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part + torch.sum(0.5 * interaction_part_2 - interaction_part_1)
        return output
    def forward(self, x):
        return self.fm_layer(x)

class Model_Loss(torch.nn.Module):
    """
    Implementation of the customized Loss Function
    """
    def __init__(self):
        super(Model_Loss, self).__init__()
        
    def forward(self, q_eval, q_target):
        loss = mseloss(q_eval, q_target)
        return loss 

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # self.fm = FM_Layer()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        # self.loss_func = nn.MSELoss(self.eval_net, self.target_net)
        self.loss_func = Model_Loss()                       # the target label is not one-hotted
        
        self.transfer_matrix = np.zeros((TIME_BUCKET_NUM * BASE_STATE_NUM, BASE_STATE_NUM))
        self.saveFile = '1.txt'
        self.lossValue = 0
        self.q_eval = 0
        self.q_target = 0
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # print(x)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        # print("action:", action)
        return action

    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

        
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        self.q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        self.q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        
        loss = self.loss_func(self.q_eval, self.q_target)
        self.lossValue = float(loss.data.numpy())
        
    def get_transfer_vector(self, base_state_id, time_slot_id):
        transfer_vector = self.eval_net( torch.from_numpy( np.array([ base_state_id, time_slot_id]) ).type(torch.FloatTensor) ).data.numpy()
        transfer_vector = transfer_vector - np.min(transfer_vector)
        transfer_vector = transfer_vector / np.sum(transfer_vector)
        return transfer_vector
    
    def update_transfer_matrix(self):
        for time_slot in range( TIME_BUCKET_NUM):
            for base_s in range( BASE_STATE_NUM):
                transfer_vector = self.get_transfer_vector(base_s, time_slot)
                self.transfer_matrix[time_slot * BASE_STATE_NUM + base_s] = transfer_vector
    
    def save(self):
        for time_slot in range( TIME_BUCKET_NUM):
            for base_s in range( BASE_STATE_NUM):
                outStr = "%s|%s|%s\t%s" %(time_slot//24 + 1, time_slot % 24 + 1, base_s + 1, str(np.around(self.transfer_matrix[time_slot * BASE_STATE_NUM + base_s],decimals=2) ))
                # print(outStr)
                commonF.append_str(self.saveFile,outStr)