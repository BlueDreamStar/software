# -*- coding: UTF-8 -*-
import logging
import numpy as np
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120] #状态空间
        
        tempX=np.arange(60,450,40)
        tempY=np.arange(40,520,40)
        catTempX=np.r_[tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX,tempX]
        catTempY=np.c_[tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY,tempY].reshape(120)
        
        self.x=list(catTempX)
        self.y=list(catTempY)
        # self.x=[140,220,300,380,460,140,300,460]
        # self.y=[250,250,250,250,250,150,150,150]
        self.terminate_states = dict()  #终止状态为字典格式
        # self.terminate_states[6] = 1
        # self.terminate_states[7] = 1
        # self.terminate_states[8] = 1

        self.error_times = 0
        self.actions = ['a','b','c','d','e']
        # self.actions = ['n','e','s','w']

        self.rewards = dict();        #回报的数据结构为字典
        self.rewards['116_a'] = 1
        self.rewards['117_a'] = 1
        self.rewards['118_a'] = 1
        self.rewards['119_a'] = 1
        self.rewards['120_a'] = 1 
        self.rewards['1_b'] = 1
        self.rewards['2_b'] = 1
        self.rewards['3_b'] = 1
        self.rewards['4_b'] = 1
        self.rewards['5_b'] = 1
        self.rewards['6_b'] = 1
        self.rewards['7_b'] = 1
        self.rewards['8_b'] = 1
        self.rewards['9_b'] = 1
        self.rewards['10_b'] = 1
        self.rewards['21_b'] = 1
        self.rewards['22_b'] = 1
        self.rewards['23_b'] = 1
        self.rewards['24_b'] = 1
        self.rewards['25_b'] = 1
        self.rewards['31_b'] = 1
        self.rewards['32_b'] = 1
        self.rewards['33_b'] = 1
        self.rewards['34_b'] = 1
        self.rewards['35_b'] = 1
        self.rewards['36_b'] = 1
        self.rewards['37_b'] = 1
        self.rewards['38_b'] = 1
        self.rewards['39_b'] = 1
        self.rewards['40_b'] = 1
        self.rewards['51_b'] = 1
        self.rewards['52_b'] = 1
        self.rewards['53_b'] = 1
        self.rewards['54_b'] = 1
        self.rewards['55_b'] = 1
        self.rewards['56_e'] = 1
        self.rewards['57_e'] = 1
        self.rewards['58_e'] = 1
        self.rewards['59_e'] = 1
        self.rewards['60_e'] = 1
        self.rewards['61_d'] = 1
        self.rewards['62_d'] = 1
        self.rewards['63_d'] = 1
        self.rewards['64_d'] = 1
        self.rewards['65_d'] = 1
        self.rewards['66_a'] = 1
        self.rewards['67_a'] = 1
        self.rewards['68_a'] = 1
        self.rewards['69_a'] = 1
        self.rewards['70_a'] = 1
        self.rewards['71_b'] = 1
        self.rewards['72_b'] = 1
        self.rewards['73_b'] = 1
        self.rewards['74_b'] = 1
        self.rewards['75_b'] = 1
        self.rewards['76_b'] = 1
        self.rewards['77_b'] = 1
        self.rewards['78_b'] = 1
        self.rewards['79_b'] = 1
        self.rewards['80_b'] = 1
        self.rewards['81_e'] = 1
        self.rewards['82_e'] = 1
        self.rewards['83_e'] = 1
        self.rewards['84_e'] = 1
        self.rewards['85_e'] = 1
        self.rewards['86_c'] = 1
        self.rewards['87_c'] = 1
        self.rewards['88_c'] = 1
        self.rewards['89_c'] = 1
        self.rewards['90_c'] = 1
        self.rewards['96_b'] = 1
        self.rewards['97_b'] = 1
        self.rewards['98_b'] = 1
        self.rewards['99_b'] = 1
        self.rewards['100_b'] = 1
        self.rewards['101_c'] = 1
        self.rewards['102_c'] = 1
        self.rewards['103_c'] = 1
        self.rewards['104_c'] = 1
        self.rewards['105_c'] = 1
        self.rewards['106_a'] = 1
        self.rewards['107_a'] = 1
        self.rewards['108_a'] = 1
        self.rewards['109_a'] = 1
        self.rewards['110_a'] = 1
        self.rewards['111_b'] = 1
        self.rewards['112_b'] = 1
        self.rewards['113_b'] = 1
        self.rewards['114_b'] = 1
        self.rewards['115_b'] = 1

        self.t = dict();             #状态转移的数据格式为字典
        self.timebucket = 24
        for t in range(self.timebucket - 1):
            for act_1 in range(len(self.actions)):
                for act_2 in range(len(self.actions)):
                    dicStr = '%s_%s' %( t*5+1 + act_1, self.actions[act_2])
                    nextState = t*5+1+ act_2 + 5
                    self.t[dicStr] = nextState
                    # print(dicStr)
        ## 末尾填充
        for act_1 in range(len(self.actions)):
            for act_2 in range(len(self.actions)): 
                dicStr = '%s_%s' %( (self.timebucket - 1)*5+1 + act_1, self.actions[act_2])
                nextState = 1+ act_2
                self.t[dicStr] = nextState
                # print(dicStr)
        # print(self.t)
            # dics = ''
        # self.t['1_a'] = 6
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
        # print("here")
        state = self.state
        # print("state",state)
        if state in self.terminate_states:
            return state, 0, self.error_times, {}
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值
        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state
        # is_terminal = False
        # if next_state in self.terminate_states:
        #     is_terminal = True
        if key not in self.rewards:
            self.error_times += 1
            r = 0.0
        else:
            self.error_times = 0
            r = self.rewards[key]
        return next_state, r, self.error_times, {}
    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state
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
            #创建第一个骷髅
            # self.kulo1 = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(140,150))
            # self.kulo1.add_attr(self.circletrans)
            # self.kulo1.set_color(0,0,0)
            # #创建第二个骷髅
            # self.kulo2 = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(460, 150))
            # self.kulo2.add_attr(self.circletrans)
            # self.kulo2.set_color(0, 0, 0)
            #创建金条
            # self.gold = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(300, 150))
            # self.gold.add_attr(self.circletrans)
            # self.gold.set_color(1, 0.9, 0)
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
            # self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')







