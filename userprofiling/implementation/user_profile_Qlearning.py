# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-02-18 17:06:38
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-02-18 17:07:06
# -*- coding: UTF-8 -*-
import sys
import gym
import random
random.seed(0)
import time
import matplotlib.pyplot as plt
from user_profile_constant import *
from user_profile_common_function import *

#  贪婪策略
def greedy(qfunc, state):
    amax = 0
    key = "%d_%s" % (state, ACTIONS[0])
    qmax = qfunc[key]
    for i in range(len(ACTIONS)):  # 扫描动作空间得到最大动作值函数
        key = "%d_%s" % (state, ACTIONS[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    return ACTIONS[amax]


#######epsilon贪婪策略
def epsilon_greedy(qfunc, state):
    amax = 0
    key = "%d_%s"%(state, ACTIONS[0])
    qmax = qfunc[key]
    for i in range(len(ACTIONS)):    #扫描动作空间得到最大动作值函数
        key = "%d_%s"%(state, ACTIONS[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    #概率部分
    pro = [0.0 for i in range(len(ACTIONS))]
    pro[amax] += 1-EPSILON
    for i in range(len(ACTIONS)):
        pro[i] += EPSILON/len(ACTIONS)

    ##选择动作
    r = random.random()
    s = 0.0
    for i in range(len(ACTIONS)):
        s += pro[i]
        if s>= r: return ACTIONS[i]
    return ACTIONS[len(ACTIONS)-1]

def qlearning(qfunc, alpha, env, REAL_TRACE):
    #初始化初始状态
    s = env.reset()
    a = ACTIONS[int(random.random()*len(ACTIONS))]
    count = 0
    while count < TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS - 1):
        # key = "%d_%s"%(s, a)
        #与环境进行一次交互，从环境中得到新的状态及回报
        s1,info  = env.step(a)
        s_num = s[1] * BASE_STATE_NUM + s[0]
        s1_num = s1[1] * BASE_STATE_NUM + s1[0]

        key = "%d_%s"%(s_num, a)
        if REAL_TRACE[(count+1)%TIME_BUCKET_NUM] == s1[0]:
            r = 1
        else:
            r = -0.5
        # r = rewards.getReward(s1_num)
        key1 = ""
        #s1处的最大动作
        a1 = greedy(qfunc, s1_num)
        key1 = "%d_%s"%(s1_num, a1)
        #利用qlearning方法更新值函数
        qfunc[key] = qfunc[key] + alpha*(r + GAMMA * qfunc[key1]-qfunc[key])
        #转到下一个状态
        s = s1;
        a = epsilon_greedy(qfunc, s1_num)
        count += 1
    return qfunc