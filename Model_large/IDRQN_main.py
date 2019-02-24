# Xinwu Qian 2019-02-06

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

# Setting the training parameters
batch_size = config.TRAIN_CONFIG['batch_size']
trace_length = config.TRAIN_CONFIG['trace_length']  # How long each experience trace will be when training
update_freq = config.TRAIN_CONFIG['update_freq']  # How often to perform a training step.
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


tau = 0.3

# --------------Simulation initialization
sys_tracker = system_tracker()
sys_tracker.initialize(distance, travel_time, arrival_rate, int(taxi_input), N_station)
env = te.taxi_simulator(arrival_rate, OD_mat, distance, travel_time, taxi_input)
env.reset()
print('System Successfully Initialized!')
# ------------------Train the network-----------------------


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
    linear_model=network.linear_model(N_station)
    for station in range(N_station):
        stand_agent.append(DRQN_agent.drqn_agent(str(station), N_station, h_size, tau, sess, batch_size, trace_length,
                                                 prioritized=prioritized, is_gpu=use_gpu))

    global_init = tf.global_variables_initializer()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    # writer.close()
    sess.run(global_init)

    for i in range(num_episodes):
        episodeBuffer = [[] for station in range(N_station)]
        sys_tracker.new_episode()
        # Reset environment and get first new observation
        env.reset()
        # return the current state of the system
        sP, tempr, featurep,score = env.get_state()
        # process the state into a list
        s = network.processState(sP, N_station)
        feature=featurep

        rAll = 0
        j = 0
        total_serve = 0
        total_leave = 0
        # We train one station in one single episode, and hold it unchanged for other stations, and we keep rotating.


        while j < max_epLength:

            j += 1

            # for all the stations, act greedily
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a = [-1] * N_station

            if softmax_action == True:  # use softmax
                for station in range(N_station):
                    if env.taxi_in_q[station]:
                        Qdist = stand_agent[station].predict_softmax(s)
                        Qprob = network.compute_softmax(Qdist);
                        a1_v = np.random.choice(Qprob[0], p=Qprob[0])
                        a1 = np.argmax(Qprob[0] == a1_v)
                        a[station] = a1  # action performed by DRQN


            else:  # use e-greedy
                predict_score = sess.run(linear_model.linear_Yh, feed_dict={linear_model.linear_X: [feature]})
                for station in range(N_station):
                    if env.taxi_in_q[station]:
                        a1 = stand_agent[station].predict(s,predict_score[0],e,station)
                        a[station] = a1  # action performed by DRQN
                        if a[station] == N_station:
                            a[station] = station

            if config.TRAIN_CONFIG['use_tracker']:
                sys_tracker.record(s, a)

            # move to the next step based on action selected
            ssp, lfp = env.step(a)
            total_serve += ssp
            total_leave += lfp

            # get state and reward

            s1P, r, featurep,score = env.get_state()
            s1 = network.processState(s1P, N_station)

            total_steps += 1

            if total_steps > pre_train_steps and j > warmup_time:
                # start training here
                if e > endE:
                    e -= stepDrop
            # episode buffer
            # we don't store the initial 200 steps of the simulation, as warm up periods
            if j > warmup_time:
                t1 = time.time()
                for station in range(N_station):
                    # we penalize the reward to motivate long term benefits
                    # newr=r+rp[a[station]]  #system reward + shared reward
                    newr = r
                    # only record the buffer for the chosen agent
                    if a[station] == -1:
                        newr = 0
                        a[station] = N_station
                    episodeBuffer[station].append(np.reshape(np.array([s, a[station], newr, s1,feature,score,featurep]), [1,
                                                                                                   7]))  # use a[nn] for action taken by that specific agent
                    if total_steps > pre_train_steps and j > warmup_time:
                        # start training here
                        # We train the selected agent
                        if total_steps % (update_freq) == 0:
                            stand_agent[station].update_target_net()  # soft update target network
                            # Reset the recurrent layer's hidden state
                            if prioritized:
                                tree_idx, trainBatch, ISWeights = stand_agent[station].buffer.sample(batch_size,
                                                                                                trace_length)
                                abs_error = stand_agent[station].per_train(trainBatch, trace_length, batch_size, ISWeights)
                                stand_agent[station].buffer.batch_update(tree_idx, abs_error)

                            else:
                                trainBatch = stand_agent[station].buffer.sample(batch_size,
                                                                           trace_length)  # Get a random batch of experiences.
                                # Below we perform the Double-DQN update to the target Q-values
                                stand_agent[station].train(trainBatch, trace_length, batch_size,linear_model,e,station,N_station)
                                #just train once
                                if station%5==0 and config.TRAIN_CONFIG['use_linear']:
                                    linear_buffer= stand_agent[station].buffer.sample(batch_size,
                                                                                              trace_length)  # Get a random batch of experiences.

                                    sess.run(linear_model.linear_update,
                                                  feed_dict={linear_model.linear_X: np.vstack(trainBatch[:, 4]),
                                                             linear_model.linear_Y: np.vstack(trainBatch[:, 5])})



                # update reward after the warm up period
                rAll += r
                if total_steps > pre_train_steps and total_steps % (update_freq) == 0 and silent == 0:
                    print('training time:', time.time() - t1)
            # swap state
            s = s1
            sP = s1P
            feature=featurep

        # Add the episode to the experience buffer
        for station in range(N_station):
            bufferArray = np.array(episodeBuffer[station])
            tempArray = []
            # now we break this bufferArray into tiny steps, according to the step length
            # lets allow overlapping for halp of the segment
            # e.g., if trace_length=20, we store [0-19] and [10-29]....keep this
            for point in range(2 * (len(bufferArray) + 1 - trace_length) // trace_length):
                stand_agent[station].remember(
                    bufferArray[(point * (trace_length // 2)):(point * (trace_length // 2) + trace_length)])

        jList.append(j)
        rList.append(rAll)  # reward in this episode
        print('Episode:', i, ', totalreward:', rAll, ', total serve:', total_serve, ', total leave:', total_leave,
              ', terminal_taxi_distribution:', [len(v) for v in env.taxi_in_q], ', terminal_passenger:',
              [len(v) for v in env.passenger_qtime], e)
        reward_out.write(str(i) + ',' + str(rAll) + '\n')


        # Periodically save the model.
        # if i % 100 == 0 and i != 0:
        #     saver.save(sess, path + '/model-' + str(i) + '.cptk')
        #     print("Saved Model")
        # if len(rList) % summaryLength == 0 and len(rList) != 0:
        #     print(total_steps, np.mean(rList[-summaryLength:]), e)
        #             saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
# summaryLength,h_size,sess,mainQN,time_per_step)
# saver.save(sess,path+'/model-'+str(i)+'.cptk')
reward_out.close()
sys_tracker.save('IDRQN')