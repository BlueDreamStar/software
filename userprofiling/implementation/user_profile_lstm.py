# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-01-06 18:19:56
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   ppvsgg
# @Last Modified time: 2019-02-18 17:07:47
from user_profile_common_function import *
from user_profile_constant import *
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.functional as F

def get_train_loader(REAL_TRACE):
    print("DataLoadStart")
    time_arrays = np.arange(TIME_BUCKET_NUM)
    times = []
    for i in range(TRAIN_WINDOW_NUMS):
        times.append(time_arrays)
    times = np.array(times).reshape(1,-1)

    ### transfer data 
    tempArray = REAL_TRACE[:TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS)].reshape(1,-1)
    data_X = np.r_[tempArray,times][:,:len(times[0]) - 1].T
    data_Y = REAL_TRACE[1:TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS)].reshape(1,-1).T

    data = torch.cat(( torch.from_numpy(data_X), torch.from_numpy(data_Y)), dim=1)    # concat indexes with input data to utilize the index
    data = data[np.newaxis, :, :]
    torch_dataset = Data.TensorDataset(data) # transform tensor to tensor dataset
    train_loader = Data.DataLoader(             # transform tensor dataset to data loader
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, #Thread to use
    )
    return train_loader

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE_LSTM,
            hidden_size=32,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 10) ## 64(label) + 1(time) + 1(gen) + 241(POI) 
        
    def forward(self, x, h_state):
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state