# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-01-06 21:05:42
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-02-18 17:08:39
import numpy as np
from user_profile_constant import *

class Rewards(object):
    def __init__(self, loadingFileDir):
        print("""
        #############################
            Begin to load rewards
        #############################
        """)
        ### Load file
        ### col_1: user_id        mapping id, 5000
        ### col_2: hour
        ### col_3: time_offset    unit:seconds  from:2018.04.01 00:00:00
        ### col_4: day_type_int   weekday:0  weekend:1
        ### col_5: region_x       longitude
        ### col_6: region_y       latitude
        ### col_7: reg_id_dict    mapping id, 51296
        ### col_8: clt_id         cluster id
        rewardFile = np.loadtxt(loadingFileDir,int)
        self.saveFile = "%s_Reward" %(loadingFileDir)
        self.rewardSet = {}
        self.real_trace_reward_set = {}

        tempN = 0
        lastTimeSlot, lastClusterID = -1, -1
        
        for n in range(len(rewardFile)):
            tempN += 1
            # tempKey = BASE_STATE_NUM * rewardFile[n,1] + rewardFile[n,7]
            tempKey = BASE_STATE_NUM * ( (rewardFile[n,2]%TIME_WINDOW)//TIME_PERIOD) + rewardFile[n,7]

            ### Set the reward
            if not self.rewardSet.has_key(tempKey):
                self.rewardSet[tempKey] = 1
                lastTimeSlot, lastClusterID = (rewardFile[n,2]%TIME_WINDOW)//TIME_PERIOD , rewardFile[n,7]
            # else:
            #     if lastTimeSlot == ((rewardFile[n,2]%TIME_WINDOW)//TIME_PERIOD) and (rewardFile[n,7] == lastClusterID):
            #         lastTimeSlot, lastClusterID = (rewardFile[n,2]%TIME_WINDOW)//TIME_PERIOD , rewardFile[n,7]
            #         continue
            #     self.rewardSet[tempKey] += 1
            #     lastTimeSlot, lastClusterID = (rewardFile[n,2]%TIME_WINDOW)//TIME_PERIOD , rewardFile[n,7]
        # print(self.rewardSet)
        
        self.reward_matrix = np.zeros((TIME_BUCKET_NUM * BASE_STATE_NUM, BASE_STATE_NUM))
        self.update_reward_matrix()
        
        tempIndexes = np.arange(TIME_BUCKET_NUM) * BASE_STATE_NUM
        self.real_trace = np.argmax(self.reward_matrix, axis=1)[tempIndexes]
        
        for time_slot in range( TIME_BUCKET_NUM):
            cur_state = self.real_trace[time_slot]
            # for base_s in range( BASE_STATE_NUM):
            self.real_trace_reward_set[ time_slot * BASE_STATE_NUM + cur_state] = 1
        
        print(self.real_trace)
        # print( np.argmax(self.reward_matrix, axis=1).shape )
        
        print("""
        #############################
            End of load rewards
        #############################
        """)
        self.saveReward()
        
    def getReward(self, stateValue):
        if not self.rewardSet.has_key(stateValue):
            return DEFAULT_REWARD
        else:
            return self.rewardSet[stateValue]
        
    def get_reward_real(self, stateValue):
        if not self.real_trace_reward_set.has_key(stateValue):
            return DEFAULT_REWARD
        else:
            return self.real_trace_reward_set[stateValue]
        
    def update_reward_matrix(self):
        for time_slot in range( TIME_BUCKET_NUM):
            for base_s in range( BASE_STATE_NUM):
                reward_vector = [self.getReward( ((time_slot + 1)% 24) * BASE_STATE_NUM + i) for i in range(BASE_STATE_NUM) ]
                self.reward_matrix[time_slot * BASE_STATE_NUM + base_s] = reward_vector

    def saveReward(self):
        for time_slot in range( TIME_BUCKET_NUM):
            for curState in range( BASE_STATE_NUM):
                reward_vector = self.reward_matrix[time_slot * BASE_STATE_NUM + curState] 
                outStr = "%s|%s|%s\t%s" %(time_slot//24 + 1, time_slot % 24 + 1, curState + 1, str(reward_vector) )
                self.appendStr(outStr)
                
    def appendStr(self, string):
        open("%s" %self.saveFile, "a").write(string+"\n")