# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2018-12-28 16:25:13
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-01-07 21:32:56
import logging
import math
import gym
import numpy as np
import random
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

#### CONSTANT VALUE
DEFAULT_REWARD = 0 # If the state is not stored in the dictionary, the reward will be the default value;
BASE_STATE_NUM = 10   # LDA Cluster Nums;
TIME_BUCKET_NUM = 7 * 24 # One day was divided into 24 time buckets.

class UserProfileEnv(gym.Env):
    """
    Description:

        A user could transfer from one state to another state.
        For example, a person may go to work at 8 a.m. from home and go to restaurant at 12:30 p.m.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        State: Home, Office, Restaurant, Bar etc.
        Time:  Hour / Week
        
    Actions:
        Type: Discrete(Number of States)
        Action: Change to another state
        
    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:

    Episode Termination:
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.action_space = spaces.Discrete(5)
        self.steps_beyond_done = None
        self.states = spaces.Discrete(BASE_STATE_NUM * TIME_BUCKET_NUM)
        
        tempX=np.arange(60,450,40)
        tempY=np.arange(40,520,40)
        catTempX=np.r_[tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX]
        catTempY=np.c_[tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY].reshape(120)
        
        self.x=list(catTempX)
        self.y=list(catTempY)
        self.terminate_states = dict()  #终止状态为字典格式

        self.error_times = 0
        self.actions = np.arange(BASE_STATE_NUM)
        
        self.t = dict();             #状态转移的数据格式为字典
        for t in range(TIME_BUCKET_NUM - 1):
            for act_1 in range(len(self.actions)):
                for act_2 in range(len(self.actions)):
                    dicStr = '%s_%s' %( t*BASE_STATE_NUM+1 + act_1, self.actions[act_2])
                    nextState = t*BASE_STATE_NUM+1+ act_2 + BASE_STATE_NUM
                    self.t[dicStr] = nextState
                    # print(dicStr)
        ## 末尾填充
        for act_1 in range(len(self.actions)):
            for act_2 in range(len(self.actions)): 
                dicStr = '%s_%s' %( (TIME_BUCKET_NUM - 1)*BASE_STATE_NUM+1 + act_1, self.actions[act_2])
                nextState = 1+ act_2
                self.t[dicStr] = nextState
                # print(dicStr)
        # print(self.t)
            # dics = ''
        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None
    def getTkey(self, tkey):
        print(tkey)
        return self.t[key]

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions
    
    def getTerminate_states(self):
        return self.terminate_states
    
    def setAction(self,s):
        self.state=s

    def step(self, action):
        #系统当前状态
        state = self.state
        if state in self.terminate_states:
            return self.decode(state), 1, {}
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值
        #状态转移
        # if key in self.t:
        next_state = self.t[key]
        # else:
            # next_state = state
        self.state = next_state
        return self.decode(self.state), self.state, {}

    def encode(self, curState, timeSlot):
        i = timeSlot * BASE_STATE_NUM + curState 
        return i

    def decode(self, i):
        out = []
        out.append(i % BASE_STATE_NUM)
        i = i // BASE_STATE_NUM
        out.append(i)
        return out  ### BASE_STATE, TIME_SLOT
        # return reversed(out)

    def reset(self):
        self.state = int(random.random() * BASE_STATE_NUM * TIME_BUCKET_NUM) + 1  ### STATE ID STAET FROM NUM 1
        self.error_times = 0
        return self.decode(self.state)
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 480
        screen_height = 520

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界
            # for 
            self.line1 = rendering.Line((40,20),(40,500))
            self.line2 = rendering.Line((80, 20), (80, 500))
            self.line3 = rendering.Line((120, 20), (120, 500))
            self.line4 = rendering.Line((160, 20), (160, 500))
            self.line5 = rendering.Line((200, 20), (200, 500))
            self.line6 = rendering.Line((240, 20), (240, 500))
            self.line7 = rendering.Line((280, 20), (280, 500))
            self.line8 = rendering.Line((320, 20), (320, 500))
            self.line9 = rendering.Line((360, 20), (360, 500))
            self.line10 = rendering.Line((400, 20), (400, 500))
            self.line11 = rendering.Line((440, 20), (440, 500))
            self.line12 = rendering.Line((40, 20), (440, 20))
            self.line13 = rendering.Line((40, 60), (440, 60))
            self.line14 = rendering.Line((40,100),(440,100))
            self.line15 = rendering.Line((40, 140), (440, 140))
            self.line16 = rendering.Line((40, 180), (440, 180))
            self.line17 = rendering.Line((40, 220), (440, 220))
            self.line18 = rendering.Line((40, 260), (440, 260))
            self.line19 = rendering.Line((40, 300), (440, 300))
            self.line20 = rendering.Line((40, 340), (440, 340))
            self.line21 = rendering.Line((40, 380), (440, 380))
            self.line22 = rendering.Line((40, 420), (440, 420))
            self.line23 = rendering.Line((40, 460), (440, 460))
            self.line24 = rendering.Line((40, 500), (440, 500))
            #创建机器人
            self.robot= rendering.make_circle(15)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)
            self.line13.set_color(0, 0, 0)
            self.line14.set_color(0, 0, 0)
            self.line15.set_color(0, 0, 0)
            self.line16.set_color(0, 0, 0)
            self.line17.set_color(0, 0, 0)
            self.line18.set_color(0, 0, 0)
            self.line19.set_color(0, 0, 0)
            self.line20.set_color(0, 0, 0)
            self.line21.set_color(0, 0, 0)
            self.line22.set_color(0, 0, 0)
            self.line23.set_color(0, 0, 0)
            self.line24.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.line13)
            self.viewer.add_geom(self.line14)
            self.viewer.add_geom(self.line15)
            self.viewer.add_geom(self.line16)
            self.viewer.add_geom(self.line17)
            self.viewer.add_geom(self.line18)
            self.viewer.add_geom(self.line19)
            self.viewer.add_geom(self.line20)
            self.viewer.add_geom(self.line21)            
            self.viewer.add_geom(self.line22)
            self.viewer.add_geom(self.line23)
            self.viewer.add_geom(self.line24)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')