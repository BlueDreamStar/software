# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-01-06 21:05:42
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2019-04-23 16:30:57
import numpy as np
from user_profile_constant import *

class Traces(object):
    def __init__(self, loadingFileDir):
        print("""
        #############################
            Begin to load traces
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
        
        prepareData = rewardFile[:,[2,7]]
        prepareData[:,0] = prepareData[:,0]//TIME_PERIOD
        
        # self.rewardSet = np.zeros( TIME_BUCKET_NUM, int)
        self.real_trace = np.zeros(TIME_BUCKET_NUM * TRAIN_WINDOW_NUMS, int)
#         self.saveFile = "%s_Reward" %(loadingFileDir)
#         tempN = 0
        
        curTimeSlot, curClusterID = -1, -1
        lastTimeSlot, lastClusterID = -1, -1
        
        curTimeSlot, curClusterID = prepareData[0,0] , prepareData[1,1]
        
        ## update first location and his pre locations
        self.real_trace[0:curTimeSlot + 1] = curClusterID
        lastTimeSlot, lastClusterID = curTimeSlot, curClusterID
        
        # print("curTimeSlot, curClusterID,", curTimeSlot, curClusterID)
        for n in range(1, len(rewardFile)):
            ## get out circle and update the tails
            if  (rewardFile[n,2] > TIME_WINDOW * TRAIN_WINDOW_NUMS):
                print("lastClusterID,",prepareData[n,0], lastClusterID)
                self.real_trace[lastTimeSlot:TIME_BUCKET_NUM * TRAIN_WINDOW_NUMS] = lastClusterID
                break
            ## update interval locations
            curTimeSlot, curClusterID = prepareData[n,0], prepareData[n,1]
            # print("curTimeSlot, curClusterID,", curTimeSlot, curClusterID)
            self.real_trace[lastTimeSlot + 1: curTimeSlot + 1] = curClusterID
            lastTimeSlot, lastClusterID = curTimeSlot, curClusterID

        print(len(self.real_trace))
        tempShape = self.real_trace.reshape( TRAIN_WINDOW_NUMS,168)
        for i in range(TRAIN_WINDOW_NUMS):
            print(tempShape[i])
        print(self.real_trace[len(self.real_trace)-168:])
        print("""
        #############################
            End of load traces
        #############################
        """)
        # self.saveReward()