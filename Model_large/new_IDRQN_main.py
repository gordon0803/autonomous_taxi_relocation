# Xinwu Qian 2019-02-06
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# This implements independent q learning approach
use_gpu = 1
import os
import config
import taxi_env as te
import taxi_util as tu
import time
from datetime import datetime
import pickle
import tensorflow as tf
import numpy as np
import network
import DRQN_agent
from system_tracker import system_tracker
import bandit
np.set_printoptions(precision=2)
if use_gpu == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# force on gpu
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True

reward_out = open('log/IDRQN_reward_log_' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.csv', 'w+')

with open('simulation_input.dat', 'rb') as fp:
    simulation_input = pickle.load(fp)

# ------------------Parameter setting-----------------------
N_station = simulation_input['N_station']
OD_mat = simulation_input['OD_mat']
distance = simulation_input['distance']
travel_time = simulation_input['travel_time']
arrival_rate = simulation_input['arrival_rate']
taxi_input = simulation_input['taxi_input']
exp_dist=simulation_input['exp_dist']

# Setting the training parameters
batch_size = config.TRAIN_CONFIG['batch_size']
trace_length = config.TRAIN_CONFIG['trace_length']  # How long each experience trace will be when training
update_freq = config.TRAIN_CONFIG['update_freq']  # How often to perform a training step.
lstm_units=config.TRAIN_CONFIG['lstm_unit']
e_threshold=config.TRAIN_CONFIG['elimination_threshold']
y = config.TRAIN_CONFIG['y']  # Discount factor on the target Q-values
startE = config.TRAIN_CONFIG['startE']  # Starting chance of random action
endE = config.TRAIN_CONFIG['endE']  # Final chance of random action
anneling_steps = config.TRAIN_CONFIG['anneling_steps']  # How many steps of training to reduce startE to endE.
num_episodes = config.TRAIN_CONFIG['num_episodes']  # How many episodes of game environment to train network with.
load_model = config.TRAIN_CONFIG['load_model']  # Whether to load a saved model.
warmup_time = config.TRAIN_CONFIG['warmup_time'];
path = "./drqn"  # The path to save our model to.
h_size = config.TRAIN_CONFIG['h_size']
max_epLength = config.TRAIN_CONFIG['max_epLength']
pre_train_steps = max_epLength * 10  # How many steps of random actions before training begins.
softmax_action = config.TRAIN_CONFIG['softmax_action']
silent = config.TRAIN_CONFIG['silent']  # do not print training time
prioritized = config.TRAIN_CONFIG['prioritized']


tau = 0.01

# --------------Simulation initialization
sys_tracker = system_tracker()
sys_tracker.initialize(distance, travel_time, arrival_rate, int(taxi_input), N_station, num_episodes, max_epLength)
env = te.taxi_simulator(arrival_rate, OD_mat, distance, travel_time, taxi_input)
env.reset()
print('System Successfully Initialized!')
# ------------------Train the network-----------------------

#--------------Output record-------------------#
outf=open('temp_record.txt','w')
# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# network number
nn = 0
# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

linucb_agent=bandit.linucb_agnet(N_station,N_station*4)
# # this step loads the model from the model that has been saved
# if load_model == True:
#     print('Loading Model...')
#     ckpt = tf.train.get_checkpoint_state(path)
#     saver.restore(sess, ckpt.model_checkpoint_path)

# this example equals target network to the original network after every few episodes
# we may want to modify this

with tf.Session(config=config1) as sess:
    # one DRQN per station is needed, different network requires a different scope (name)
    stand_agent = []
    # targetOps=[]

    agent=DRQN_agent.drqn_agent_efficient(N_station, h_size, lstm_units,tau, sess, batch_size, trace_length,is_gpu=use_gpu)
    agent.drqn_build()

    exp_replay=network.experience_buffer(10000) #a single buffer holds everything
    bandit_buffer=network.bandit_buffer(5000)
    global_init = tf.global_variables_initializer()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    # writer.close()1
    sess.run(global_init)

    Qp_in=[]
    Qp_value_in=[]
    Q1_in=[]
    Q2_in=[]
    Q_train=[]
    Q_input_dict = dict()
    Q_train_dict = dict()
    Qp_input_dict=dict()
    for station in range(N_station):
        Qp_in.append(agent.mainPredict[station])
        Qp_value_in.append(agent.mainQout[station])
        Qp_input_dict[agent.trainLength] = 1
        Qp_input_dict[agent.batch_size] = 1
        Q1_in.append(agent.targetZ[station])
        Q2_in.append(agent.targetQout[station])
        Q_train.append(agent.updateModel[station])



    for i in range(num_episodes):
        global_epi_buffer=[]
        global_bandit_buffer=[]
        sys_tracker.new_episode()
        # Reset environment and get first new observation
        env.reset()
        # return the current state of the system
        sP, tempr, featurep,score,tr2 = env.get_state()
        # process the state into a list
        # replace the state action with future states
        feature=featurep
        s = network.processState(sP, N_station)

        rAll = 0
        rAll_unshape=0
        j = 0
        total_serve = 0
        total_leave = 0

        buffer_count=0;
        # We train one station in one single episode, and hold it unchanged for other stations, and we keep rotating.

        tinit=time.time()
        while j < max_epLength:
            tall=time.time()
            j += 1
            agent.update_conf(1,1.5*anneling_steps)

            # for all the stations, act greedily
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a = [-1] * N_station

            if config.TRAIN_CONFIG['use_linear'] == 0:  # not using action elimination

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            a[station] = np.random.randint(0, N_station)  # random actions for each station
                else:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            a1 = agent.predict_regular(s,station)
                            a[station] = a1[0]  # action performed by DRQN
                            if a[station] == N_station:
                                a[station] = station

            else:  # use e-greedy
                #predict_score = sess.run(linear_model.linear_Yh, feed_dict={linear_model.linear_X: [feature]})
                predict_score=linucb_agent.return_upper_bound(feature)
                predict_score=(predict_score*exp_dist)/distance
                rand_num=np.random.rand(1)
                for station in range(N_station):
                    if env.taxi_in_q[station]:
                        a1 = agent.predict(s,predict_score[station,:],e,station,e_threshold,rand_num)
                        a[station] = a1  # action performed by DRQN
                        if a[station] == N_station:
                            a[station] = station

                # if total_steps % (1000) == 0 and i > 4:
                #     print('Available actions to choose from:',sum(predict_score>0.4),sum(predict_score>0.3),sum(predict_score>0.2),sum(predict_score>0.1))
                #     print(predict_score)
            if config.TRAIN_CONFIG['use_tracker']:
                sys_tracker.record(s, a)

            # move to the next step based on action selected
            ssp, lfp = env.step(a)
            total_serve += ssp
            total_leave += lfp

            # get state and reward

            s1P, r, featurep,score,r2 = env.get_state()
            s1 = network.processState(s1P, N_station)

            total_steps += 1

            if total_steps > pre_train_steps and j > warmup_time:
                # start training here
                if e > endE:
                    e -= stepDrop
            # episode buffer
            # we don't store the initial 200 steps of the simulation, as warm up periods
            newr=r*np.ones((N_station))
            for station in range(N_station):
                if a[station] == -1:
                    #newr[station]=0
                    a[station] = station

            v1=np.reshape(np.array([s, a, newr, s1,feature,score,featurep]), [1,7])
            global_epi_buffer.append(v1)
            global_bandit_buffer.append(v1)

            #exp replay
            # buffer_count+=1
            # if buffer_count>=2*trace_length:
            #     #pop the first trace length items
            #     newbufferArray=[]
            #     bufferArray=global_epi_buffer[:trace_length]
            #     for bf in range(trace_length):
            #         #recalibrate the rewards
            #         reward_vec=sum([(y**id)*global_epi_buffer[bf+id][0][2] for id in range(trace_length)])
            #         bufferArray[bf][0][2]=reward_vec
            #     exp_replay.add(bufferArray)
            #     global_epi_buffer=global_epi_buffer[trace_length:]
            #     buffer_count-=trace_length

            buffer_count+=1
            if buffer_count>=2*trace_length:
                for it in range(trace_length):
                    bufferArray=np.array(global_epi_buffer)
                    exp_replay.add(bufferArray[it:it+trace_length])
                global_epi_buffer=global_epi_buffer[trace_length:]
                buffer_count-=trace_length

            if total_steps % (500) == 0 and i>4:
                linubc_train = bandit_buffer.sample(batch_size * 20)
                linucb_agent.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])

            #use a single buffer
            if total_steps > pre_train_steps and j > warmup_time:
                #train linear multi-arm bandit first, we periodically update this (every 10*update_fequency steps)
                t1 = time.time()
                if total_steps % (update_freq) == 0:
                    agent.update_target_net()

                    trainBatch = exp_replay.sample(batch_size, trace_length)
                    # print('LINUCB update time:',time.time()-t1)
                    train_predict_score=linucb_agent.return_upper_bound_batch(trainBatch[:,6])
                    train_predict_score=train_predict_score*exp_dist
                   # train_predict_score[train_predict_score<0.312]=100
                   # train_predict_score[train_predict_score>=0.312] =0
                   #  predict_in=np.zeros((batch_size*trace_length,N_station+1))
                # predict_in[:,:-1]=train_predict_score;
                    # print('LINUCB predict time:', time.time() - t1)
                    #get targetQ
                    var = np.vstack(trainBatch[:, 3])
                    Q_input_dict[agent.scalarInput] = var
                    Q_input_dict[agent.trainLength] = trace_length
                    Q_input_dict[agent.batch_size] = batch_size
                    t1=time.time()
                    for station in range(N_station):
                        tr, t_action = agent.train_prepare(trainBatch, station)
                        tp=train_predict_score/distance[station,:]
                        af=tp<e_threshold
                        bf=tp>=e_threshold
                        tp[af]=100
                        tp[bf]=0
                        tp[:,station]=0;
                        Q_input_dict[agent.predict_score[station]] = tp

                        Q_input_dict[agent.rewards[station]] = tr

                    targetz = sess.run(Q1_in, feed_dict=Q_input_dict)

                    # print('targetz update time:',time.time()-t1)
                    #train

                    t1=time.time()
                    var=np.vstack(trainBatch[:, 0])
                    Q_train_dict[agent.scalarInput] = var
                    Q_train_dict[agent.trainLength] = trace_length
                    Q_train_dict[agent.batch_size] = batch_size
                    for station in range(N_station):
                        tr,t_action=agent.train_prepare(trainBatch, station)
                        Q_train_dict[agent.rewards[station]]= tr
                        Q_train_dict[agent.actions[station]]= t_action
                        Q_train_dict[agent.targetQ[station]]= targetz[station]
                    #train now
                    sess.run(Q_train,feed_dict=Q_train_dict)

                    # print('train time:', time.time() - t1)
                    # print('Sequential Train time:', time.time()-t1)


            rAll += r
            rAll_unshape+=r2
            # swap state
            s = s1
            sP = s1P
            feature=featurep

            #preocess bandit buffer
        for epi in range(len(global_bandit_buffer)-2*trace_length):
               # print(global_bandit_buffer[i])
            score=np.mean([global_bandit_buffer[epi+k][0][5] for k in range(2*trace_length)],axis=0)
            record=global_bandit_buffer[epi]
            record[0][5]=score; #replay the score
            bandit_buffer.add(record)



        jList.append(j)
        rList.append(rAll)  # reward in this episode
        
        sys_tracker.record_time(env)
        print('Episode:', i, ', totalreward:', rAll, ', old reward:',rAll_unshape,', total serve:', total_serve, ', total leave:', total_leave, ', total_cpu_time:',time.time()-tinit,
              ', terminal_taxi_distribution:', [len(v) for v in env.taxi_in_q], ', terminal_passenger:',
              [len(v) for v in env.passenger_qtime], e,agent.conf)
        reward_out.write(str(i) + ',' + str(rAll) + '\n')
        outf.writelines(str(i) + ',' + str(rAll) + ',' + str(total_serve) + ',' + str(total_leave) + '\n')

        # Periodically save the model.
        # if i % 100 == 0 and i != 0:
        #     saver.save(sess, path + '/model-' + str(i) + '.cptk')
        #     print("Saved Model")
        # if len(rList) % summaryLength == 0 and len(rList) != 0:
        #     print(total_steps, np.mean(rList[-summaryLength:]), e)
        #             saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
# summaryLength,h_size,sess,mainQN,time_per_step)
# saver.save(sess,path+'/model-'+str(i)+'.cptk')
outf.close()
reward_out.close()
sys_tracker.save('IDRQN')
sys_tracker.playback(-1)
