DEFAULT_REWARD = 0 # If the state is not stored in the dictionary, the reward will be the default value;
BASE_STATE_NUM = 10   # LDA Cluster Nums;
TIME_BUCKET_NUM = 7 * 24 # One day was divided into 24 time buckets.

# Hyper Parameters
DAY_SECONDS = 86400
WEEK_SECONDS = 7 * 86400
TIME_WINDOW = WEEK_SECONDS
TIME_PERIOD = TIME_WINDOW // TIME_BUCKET_NUM
TRAIN_WINDOW_NUMS = 10

### user profile main
MAX_EPISODE = 10
MEMORY_CAPACITY = 100
LEARNING_RATE = 0.01
N_ACTIONS = 10
N_STATES = 2
N_NOTES = 30

###  
BATCH_SIZE = 32
EPSILON = 0.8               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 200
Vector_N = 10
Embedding_K = 5
STEP_WINDOWS = 10
TRACE_LENGTH_STEPS = TIME_BUCKET_NUM * TRAIN_WINDOW_NUMS - TIME_BUCKET_NUM

FILE_DIR = "../datasets"
SAVE_DIR = "../outcome"
GYM_ENV_Name = 'UserProfile-v0'

# Hyper Parameters
BATCH_SIZE = 100
# TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE_LSTM = 2    # rnn input size, (cur_state, time)

import torch.nn as nn
crossentropyloss = nn.CrossEntropyLoss() 
mseloss = nn.MSELoss()   # initialization of the MSE loss function
