# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-02-18 17:08:14
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-02-18 17:08:22
import sys
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from user_profile_DQN_fm import DQN
from user_profile_constant import *
from load_trace import Traces
sys.path.append('/home/bil/wpf/lib/baseLibrary/')
import baseMethods.LDA as baseLDA
import metrics.common as commonF

class Model_Loss(torch.nn.Module):
    """
    Implementation of the customized Loss Function
    """
    def __init__(self):
        super(Model_Loss, self).__init__()
        
    def forward(self, q_eval, q_target):
        loss = mseloss(q_eval, q_target)
        return loss 
    
def trace_overlap(predict_trace, real_trace):
    # print("predict_trace, real_trace", predict_trace, real_trace)
    bool_matrix =  real_trace == predict_trace
    overlap_pro = 1.0 * len(np.where( bool_matrix == True)[0])/len(real_trace) 
    return overlap_pro

### get the userIDs
def get_user_ids(sys_avgs):
    argv_length = len(sys_avgs)

    if argv_length <= 1:
        userIDs = np.arange(1)
    elif argv_length == 2:
        userIDs = np.arange(int(sys_avgs[1]), int(sys_avgs[1]) + 1)
    else:
        userIDs = np.arange(int(sys_avgs[1]), int(sys_avgs[2]) + 1)
        
    return userIDs


### get traces by userID
def get_trace_list( userIDs):
    user_nums = len(userIDs)
    dqn_list = []
    trace_data_list = []
    REAL_TRACE_LIST = []
    predict_trace_list = []
    env_list = []

    for i in range(user_nums):
        print("""
        #############################
                user_nums: %s
        #############################
        """ %i)
        tempFile = "%s/user_%s" %(FILE_DIR, userIDs[i])
        env_list.append( (gym.make(GYM_ENV_Name)).unwrapped )
        dqn_list.append(DQN())
        trace_data_list.append(Traces(tempFile))
        REAL_TRACE_LIST.append(trace_data_list[i].real_trace)
        predict_trace_list.append(np.zeros(len(REAL_TRACE_LIST[i]),int))
        
    return dqn_list, env_list, REAL_TRACE_LIST, predict_trace_list


def env_states_actions():
    env = (gym.make(GYM_ENV_Name)).unwrapped
    return env.getStates(), env.getAction()

STATES, ACTIONS = env_states_actions()
