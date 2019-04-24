# Xinwu Qian 2019-02-06
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# This implements independent q learning approach
use_gpu = 1
import os
import config
from multiprocessing import Pool
import taxi_env as te
import scipy
import taxi_util as tu
import time
from datetime import datetime
import pickle
from collections import deque
import tensorflow as tf
import numpy as np
import network
import DRQN_agent
from system_tracker import system_tracker
import bandit

from tensorflow.python.client import timeline

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
rng_seed=config.TRAIN_CONFIG['random_seed']


#set rng seed
np.random.seed(rng_seed)


tau = 0.1

# --------------Simulation initialization
sys_tracker = system_tracker()
sys_tracker.initialize(config, distance, travel_time, arrival_rate, int(taxi_input), N_station, num_episodes, max_epLength)
env = te.taxi_simulator(arrival_rate, OD_mat, distance, travel_time, taxi_input)
env.reset()
print('System Successfully Initialized!')
# ------------------Train the network-----------------------


#--------------Output record-------------------#
outf=open('temp_record.txt','w')
# Set the rate of random action decrease.
e = startE
stepDrop = endE**(1/anneling_steps)

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# network number
nn = 0
# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

linucb_agent=bandit.linucb_agent(N_station,N_station*4)
exp_replay = network.experience_buffer(15000)  # a single buffer holds everything
bandit_buffer = network.bandit_buffer(15000)
bandit_swap_e=1;
linucb_agent_backup=bandit.linucb_agent(N_station, N_station * 4)
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

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    agent=DRQN_agent.drqn_agent_efficient(N_station, h_size, lstm_units,tau, sess, batch_size, trace_length,is_gpu=use_gpu)
    agent.drqn_build()


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
    total_train_iter=0;
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
        pres=s
        prea=np.zeros((N_station))

        within_frame_reward = 0
        frame_skipping = 1

        prediction_time=0
        targetz_time=0
        training_time=0

        rAll = 0
        rAll_unshape=0
        j = 0
        total_serve = 0
        total_leave = 0

        buffer_count=0;
        # We train one station in one single episode, and hold it unchanged for other stations, and we keep rotating.
        tinit=time.time()
        a = [st for st in range(N_station)]

        #bandit swapping scheme
        #bandit swapping scheme
        if bandit_swap_e - e >.1:  # we do swapping when $e$ got declined by 0.05 percent.
            linucb_agent=linucb_agent_backup
            linucb_agent_backup=bandit.linucb_agent(N_station, N_station * 4)
            bandit_swap_e=e
            print('we swap bandit here')
        state_predict=deque()

        initial_rnn_cstate = np.zeros((1, 1, config.TRAIN_CONFIG['lstm_unit']))
        initial_rnn_hstate = np.zeros((1, 1, config.TRAIN_CONFIG['lstm_unit']))
        while j < max_epLength:
            # agent.update_conf(1,1.5*anneling_steps)

            tall=time.time()
            j += 1
            hour=(j-1)//120
            # for all the stations, act greedily
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a = [-1] * N_station

            tempt = time.time()
            if config.TRAIN_CONFIG['use_linear'] == 0:  # not using action elimination

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            a[station] = np.random.randint(0, N_station)  # random actions for each station
                else:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            a1 = agent.predict_regular(s, station)
                            a[station] = a1[0]  # action performed by DRQN
                            if a[station] == N_station:
                                a[station] = station

            else:  # use e-greedy
                # predict_score = sess.run(linear_model.linear_Yh, feed_dict={linear_model.linear_X: [feature]})
                predict_score=linucb_agent.return_upper_bound(feature)
                predict_score=predict_score*exp_dist[hour]/distance
                invalid=predict_score<e_threshold
                valid=predict_score>=e_threshold
                rand_num=np.random.rand(1)
                state_predict.append(s)
                if len(state_predict)>1:
                    state_predict.popleft()
                if rand_num < e:
                    rnn_value = 0
                    all_actions = 0
                else:
                    rnn_value,initial_rnn_state = sess.run([agent.main_rnn_value,agent.rnn_out_state],feed_dict={agent.scalarInput: np.vstack(state_predict), agent.rnn_cstate_holder:initial_rnn_cstate,agent.rnn_hstate_holder:initial_rnn_hstate,agent.iter_holder:[np.array([e])], agent.eps_holder:[np.array([total_train_iter])], agent.trainLength: len(state_predict), agent.batch_size: 1})
                    initial_rnn_cstate=initial_rnn_state[0]
                    initial_rnn_hstate=initial_rnn_state[1]
                for station in range(N_station):
                    if env.taxi_in_q[station]:
                        a1 = agent.predict(rnn_value, predict_score[station, :], e, station, e_threshold, rand_num,
                                           valid[station, :], invalid[station, :])
                        a[station] = a1  # action performed by DRQN
                        if a[station] == N_station:
                            a[station] = station
            prediction_time += time.time() - tempt

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
                    e*=stepDrop
            # episode buffer
            # we don't store the initial 200 steps of the simulation, as warm up periods
            newr=r*np.ones((N_station))

            v1=np.reshape(np.array([s, a, newr, s1,feature,score,featurep,e,total_train_iter]), [1,9])
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
                linubc_train = bandit_buffer.sample(batch_size * 40)
                linucb_agent.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])
                linucb_agent_backup.update(linubc_train[:, 4], linubc_train[:, 1], linubc_train[:, 5])


            #use a single buffer
            if total_steps > pre_train_steps and j > warmup_time:
                #train linear multi-arm bandit first, we periodically update this (every 10*update_fequency steps)
                t1 = time.time()
                if total_steps % (update_freq) == 0:
                    agent.update_target_net()
                   # train_predict_score[train_predict_score<0.312]=100
                   # train_predict_score[train_predict_score>=0.312] =0
                   #  predict_in=np.zeros((batch_size*trace_length,N_station+1))
                # predict_in[:,:-1]=train_predict_score;
                    # print('LINUCB predict time:', time.time() - t1)
                    #get targetQ
                    total_train_iter+=1/5000
                    trainBatch_list = [exp_replay.sample(batch_size, trace_length) for st in range(N_station)]


                    for station in range(N_station):
                        trainBatch= trainBatch_list[station]
                        # generate the linucb score for each batch
                        # train_predict_score= train_predict_score_list[(station)*batch_size*trace_length:(station+1)*batch_size*trace_length,:] * exp_dist
                        train_predict_score = linucb_agent.return_upper_bound_batch(trainBatch[:, 6]) * exp_dist[hour]
                        past_train_eps=np.vstack(trainBatch[:,7])
                        past_train_iter = np.vstack(trainBatch[:, 8])
                        current_action=np.vstack(trainBatch[:,1])
                        tr, t_action = agent.train_prepare(trainBatch, station)
                        tp = train_predict_score/ distance[station, :]
                        af = tp < e_threshold
                        bf = tp >= e_threshold
                        tp[af] = 100
                        tp[bf] = 0
                        tp[:, station] = 0
                        tempt = time.time()
                        tz=sess.run(agent.targetZ[station],feed_dict={agent.scalarInput:np.vstack(trainBatch[:, 3]),agent.iter_holder:past_train_iter, agent.eps_holder:past_train_eps,  agent.predict_score[station]:tp,agent.rewards[station]:tr,agent.trainLength:trace_length,agent.batch_size:batch_size})
                        targetz_time += time.time() - tempt
                        tempt = time.time()
                        sess.run(agent.updateModel[station], feed_dict={agent.targetQ[station]: tz, agent.rewards[station]: tr, agent.actions[station]: t_action, agent.scalarInput: np.vstack(trainBatch[:, 0]),agent.iter_holder:past_train_iter, agent.eps_holder:past_train_eps,agent.trainLength: trace_length, agent.batch_size: batch_size})
                        training_time += time.time() - tempt

                    # train
                    # sess.run(agent.updateModel[station],feed_dict={agent.targetQ[station]:tz,agent.rewards[station]:tr,agent.actions[station]:t_action,agent.scalarInput:np.vstack(trainBatch[:, 0]),agent.trainLength:trace_length,agent.batch_size:batch_size})


            rAll += r
            rAll_unshape+=r2
            # swap state
            s = s1
            sP = s1P
            feature=featurep

            #preocess bandit buffer
        future_steps=2*trace_length
        tmask = np.linspace(0, 1, num=future_steps + 1)
        pdeta=0.5;
        quantile_mask=scipy.stats.norm.cdf(scipy.stats.norm.ppf(tmask)-pdeta)
        quantile_mask = np.diff(quantile_mask) # rescale the distribution to favor risk neutral or risk-averse behavior

        for epi in range(len(global_bandit_buffer)-future_steps-1):
               # print(global_bandit_buffer[i])
            score=np.array([global_bandit_buffer[epi+k][0][5] for k in range(future_steps)]).T.dot(quantile_mask)
            record=global_bandit_buffer[epi]
            record[0][5]=score; #replay the score
            bandit_buffer.add(record)



        jList.append(j)
        rList.append(rAll)  # reward in this episode
        sys_tracker.record_time(env)
        print('Episode:', i, ', totalreward:', rAll, ', old reward:',rAll_unshape,', total serve:', total_serve, ', total leave:', total_leave, ', total_cpu_time:',time.time()-tinit,
              ', terminal_taxi_distribution:', [len(v) for v in env.taxi_in_q], ', terminal_passenger:',
              [len(v) for v in env.passenger_qtime], e,agent.conf)
        n_vars=len(tf.trainable_variables())
        print('TargetZ_time:',targetz_time,', Training time:',training_time, ', Prediction time:',prediction_time,'Number of tensorflow variables',n_vars)
        reward_out.write(str(i) + ',' + str(rAll) + '\n')

        outf.writelines(str(i)+','+str(rAll)+','+str(total_serve)+','+str(total_leave)+'\n')


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
