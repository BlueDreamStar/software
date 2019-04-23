# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-01-07 16:06:09
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2019-04-23 16:30:09
import sys
import gym
from user_profile_constant import *
from user_profile_common_function import *
from user_profile_Qlearning import *
from user_profile_lstm import *

def update_predict_trace( start_state, env, STEP_WINDOWS):
    predict_trace = np.zeros(STEP_WINDOWS, int)
    pre_s = start_state
    temp_state = env.state
    env.state = start_state[0], start_state[1]
    # for i in range(len(predict_trace)):
    for i in range(STEP_WINDOWS):
        pre_a = dqn.choose_action(pre_s)
        predict_trace[i] = pre_a
        pre_s_, info = env.step(pre_a)
        pre_s = pre_s_
    env.state = temp_state
    return predict_trace 

def reward_overlap(overlap_pro):
    if overlap_pro > 0.5:
        return overlap_pro * 2
    elif overlap_pro > 0.1:
        return overlap_pro * 2
    else:
        return 1
    
print('''
    #############################################
                    User Profiling
    #############################################

    ''')

def dqn_execute(dqn, env, REAL_TRACE, predict_trace, user_id):

    loss_func = Model_Loss()
    dqn.eval_net.register_parameter('bias', None)
    optimizer = torch.optim.Adam(dqn.eval_net.parameters(), lr=0.01)
    
    ### get the save dir
    log_dir, predict_dir, overlap_dir, max_overlap_dir = log_files_check("dqn", user_id)

    ## 多个网络参数同时执行梯度下降的方法示例
    # params = list(dqn.eval_net.parameters()) + list(dqn2.eval_net.parameters()) 
    # optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    
    max_overlap = 0
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        for step in range(TRACE_LENGTH_STEPS - 1):
            a = dqn.choose_action(s)
            # take action
            s_, info = env.step(a)
            ### computer reward
            if REAL_TRACE[(step+1)%TIME_BUCKET_NUM] == s_[0]:
                r = 1
            else:
                r = -0.5
                
            dqn.store_transition(s, a, r, s_)
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                loss = loss_func(dqn.q_eval, dqn.q_target)
                optimizer.zero_grad()
                loss.backward() #retain_graph=True
                optimizer.step()
            s = s_
            
        predict_trace =  update_predict_trace( np.array([REAL_TRACE[len(REAL_TRACE)-TIME_BUCKET_NUM], 0],int), env, (TIME_BUCKET_NUM-1))
        overlap_pro = trace_overlap(predict_trace, REAL_TRACE[len(REAL_TRACE)-(TIME_BUCKET_NUM-1):])
        if overlap_pro > max_overlap:
                max_overlap = overlap_pro
        log_str = 'userId: %s, | Ep: %s, | Loss: %s, | Lap : %s' %(user_id, i_episode, round(float(loss.data.numpy()),4), round(overlap_pro,4) )
        predict_trace_list = list(predict_trace)
        predict_str = ' '.join(map(str, predict_trace_list))
        
        print(log_str)
        commonF.append_str(log_dir,log_str)
        commonF.append_str(predict_dir,predict_str)
        commonF.append_str(overlap_dir,str(overlap_pro))
    commonF.append_str(max_overlap_dir,str(max_overlap))
     

def log_files_check(function_name, user_id):
    function_name_log_dir = "%s/%s_%s.log" %(SAVE_DIR,function_name,user_id)
    function_name_predict_dir = "%s/%s_predict_%s.txt" %(SAVE_DIR,function_name,user_id)
    function_name_overlap_dir = "%s/%s_overlap_%s.txt" %(SAVE_DIR,function_name,user_id)
    function_name_max_overlap_dir = "%s/%s_max_overlap.txt" %(SAVE_DIR,function_name)
    
    commonF.delete_file(function_name_log_dir)
    commonF.delete_file(function_name_predict_dir)
    commonF.delete_file(function_name_overlap_dir)
    commonF.delete_file(function_name_max_overlap_dir)

    return function_name_log_dir, function_name_predict_dir, function_name_overlap_dir, function_name_max_overlap_dir

    
def qlearning_execute(env, REAL_TRACE, predict_trace, user_id):
    ##### 
    log_dir, predict_dir, overlap_dir, max_overlap_dir = log_files_check("qlearning", user_id)

    terminate_states= env.getTerminate_states()
    qfunc = dict()   #行为值函数为字典
    #初始化行为值函数为0
    for s in STATES:
        for a in ACTIONS:
            key = "%d_%s"%(s,a)
            qfunc[key] = 0.0
    #训练
    max_overlap = 0
    for i_episode in range(MAX_EPISODE):
        qfunc = qlearning(qfunc, 0.2, env, REAL_TRACE)
        # 设置系统初始状态
        s0 = 1
        env.setAction(s0)
        #随机初始化
        s0 = env.reset()
        count = 0
        tempList = []
        while count < TIME_BUCKET_NUM - 1:
            s_num = s0[1] * BASE_STATE_NUM + s0[0]
            a1 = greedy(qfunc, s_num)
            s1, info = env.step(a1)
            s1_num = s1[1] * BASE_STATE_NUM + s1[0]
            tempList.append(a1)
            s0 = s1
            count += 1
            
        predict_trace_list = tempList
        predict_str = ' '.join(map(str, predict_trace_list))
        overlap_pro = trace_overlap(np.array(predict_trace_list,int), REAL_TRACE[len(REAL_TRACE)-(TIME_BUCKET_NUM-1):] )
        if overlap_pro > max_overlap:
                max_overlap = overlap_pro
        log_str = 'userId: %s, | Ep: %s, | Lap : %s' %(user_id, i_episode, round(overlap_pro,4) )
        print(log_str)
        commonF.append_str(log_dir,log_str)
        commonF.append_str(predict_dir,predict_str)
        commonF.append_str(overlap_dir,str(overlap_pro))
    commonF.append_str(max_overlap_dir,str(max_overlap))
        
    ### Q-learning save policy
    ### 学到的值函数
    policy_dir = "%s/qlearning_policy_%s" %(SAVE_DIR,user_id)
    policy_value_dir = "%s/qlearning_policy_value_%s" %(SAVE_DIR,user_id)
    
    commonF.delete_file(policy_dir)
    commonF.delete_file(policy_value_dir)

    for s in STATES:
        for a in ACTIONS:
            key = "%d_%s"%(s,a)
            tempStr = "the qfunc of key (%s) is %f" %(key, qfunc[key])
            commonF.append_str(policy_value_dir,tempStr)

    #学到的策略为：
    for i in range(len(STATES)):
        if STATES[i] in terminate_states:
            tempStr = "the state %d is terminate_states"%(STATES[i])
        else:
            tempStr = "the policy of state %d is (%s)" % (STATES[i], greedy(qfunc, STATES[i]))
        commonF.append_str(policy_dir,tempStr)
        
        
def lstm_execute(REAL_TRACE, predict_trace, user_id):
    log_dir, predict_dir, overlap_dir, max_overlap_dir = log_files_check("lstm", user_id)
    
    lstm = LSTM()
    h_state = None      # for initial hidden state
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()            # the target label is not one-hotted
    
    train_loader= get_train_loader(REAL_TRACE)
    
    # training and testing
    max_overlap = 0
    for i_episode in range(MAX_EPISODE):
        for step, (inputMat) in enumerate(train_loader):        # gives batch data
            ### only x
            inputMat = inputMat[0]
            x = Variable(inputMat[:, :, :INPUT_SIZE_LSTM].contiguous()).type(torch.FloatTensor)    # re-construct input data
            y = Variable(inputMat[:, :, INPUT_SIZE_LSTM].contiguous()).type(torch.LongTensor)  

            train_X = x[:,:TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS-1),:]
            train_y = y[:,:TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS-1)]

            output, h_state = lstm(train_X, h_state)                              # rnn output

            loss = loss_func(output[-1,:,:], train_y[-1,:])                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward(retain_graph=True)                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            
            ### testing
            pre_imput_x = x[:,TIME_BUCKET_NUM * (TRAIN_WINDOW_NUMS-1),:][:,np.newaxis,:].clone()
            pre_h_state = None
            tempList = []
            
            for i in range(TIME_BUCKET_NUM - 1):
                pre_output, pre_h_state = lstm(pre_imput_x, pre_h_state)        # rnn output
                pre_next_label = torch.argmax(pre_output,dim=2)
                pre_imput_x[:,:,0] = pre_next_label
                pre_imput_x[:,:,1] = i+1

                tempList.append(pre_next_label)
            
            predict_trace_list = tempList
            predict_str = ' '.join(map(str, predict_trace_list))
            overlap_pro = trace_overlap(np.array(predict_trace_list,int), REAL_TRACE[len(REAL_TRACE)-(TIME_BUCKET_NUM-1):] )
            if overlap_pro > max_overlap:
                max_overlap = overlap_pro
            log_str = 'userId: %s, | Ep: %s, | Step: %s, | Loss: %s, | Lap : %s' %(user_id, i_episode,step, round(float(loss.data.numpy()),4), round(overlap_pro,4) )
            print(log_str)
            
            commonF.append_str(log_dir,log_str)
            commonF.append_str(predict_dir,predict_str)
            commonF.append_str(overlap_dir,str(overlap_pro))
    commonF.append_str(max_overlap_dir,str(max_overlap))
        
### main function
if __name__ == "__main__":
    userIDs = get_user_ids(sys.argv)
    dqn_list, env_list, REAL_TRACE_LIST, predict_trace_lists = get_trace_list(userIDs)
    
    for user_index in range(len(userIDs)):
        dqn = dqn_list[user_index]
        env = env_list[user_index]
        REAL_TRACE = REAL_TRACE_LIST[user_index]
        predict_trace = predict_trace_lists[user_index]
        
        user_id = userIDs = userIDs[user_index]
        
        dqn_execute(dqn, env, REAL_TRACE, predict_trace, user_id)        
        qlearning_execute(env, REAL_TRACE, predict_trace, user_id)
        lstm_execute(REAL_TRACE, predict_trace, user_id)        
    print("main completely!")
