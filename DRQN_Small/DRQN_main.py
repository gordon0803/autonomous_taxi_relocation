#Xinwu Qian 2019-02-06

#Main file for DRQN

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from network import *


import taxi_env as te
import taxi_util as tu
import DRQN_agent
import network
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


#------------------Parameter setting-----------------------
#define the input here
N_station=20
l1=[5 for i in range(N_station)]
OD_mat=[l1 for i in range(N_station)]
distance=OD_mat
travel_time=OD_mat
arrival_rate=[0.5+i/4.0 for i in range(N_station)]
taxi_input=5

env=te.taxi_simulator(arrival_rate,OD_mat,distance,travel_time,taxi_input)
env.reset()
print('System Successfully Initialized!')

#Setting the training parameters
batch_size = 4 #How many experience traces to use for each training step.
trace_length = 8 #How long each experience trace will be when training
update_freq = 5 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 2000 #How many steps of training to reduce startE to endE.
num_episodes = 1000 #How many episodes of game environment to train network with.
pre_train_steps = 2000 #How many steps of random actions before training begins.
load_model = False #Whether to load a saved model.
warmup_time=100;
path = "./drqn" #The path to save our model to.
h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_epLength = 500 #The max allowed length of our episode.
time_per_step = 1 #Length of each step used in gif creation
summaryLength = 100 #Number of epidoes to periodically save for analysis
tau = 0.001

episode_per_agent=50; #number of training episodes for every agent.


#------------------Train the network-----------------------


# DQN combined with LSTM



# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0


#network number
nn=0
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

with tf.Session() as sess:
    # one DRQN per station is needed, different network requires a different scope (name)
    stand_agent = []
    # targetOps=[]
    for station in range(N_station):
        stand_agent.append(DRQN_agent.drqn_agent(str(station), N_station, h_size, tau,sess))


    for i in range(num_episodes):
        episodeBuffer = [[] for station in range(N_station)]

        # Reset environment and get first new observation
        env.reset()

        # return the current state of the system
        sP, tempr = env.get_state()
        # process the state into a list
        s = network.processState(sP, N_station)

        rAll = 0
        j = 0

        # We train one station in one single episode, and hold it unchanged for other stations, and we keep rotating.
        nn = i % N_station
        print(nn)

        # Reset the recurrent layer's hidden state
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        # The Q-Network
        while j < max_epLength:
            j += 1
            # for all the stations, act greedily
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a=[-1]*N_station

            if np.random.rand(1) < e or total_steps < pre_train_steps:
                state1=stand_agent[nn].get_rnn_state(s,state)
                for station in range(N_station):
                    a[station]=np.random.randint(0, N_station) #random actions for each station
            else:
                state1 = stand_agent[nn].get_rnn_state(s,state)
                for station in range(N_station):
                    a1 = stand_agent[station].predict(s,state)
                    a[station]=a1[0] #action performed by DRQN


            # move to the next step based on action selected
            env.step(a)

            # get state and reward
            s1P, r = env.get_state()
            r = r / (taxi_input * N_station + 0.0000001)
            s1 = network.processState(s1P, N_station)

            total_steps += 1

            # episode buffer
            # we don't store the initial 200 steps of the simulation, as warm up periods
            if j>warmup_time:
                for station in range(N_station):
                    episodeBuffer[station].append(np.reshape(np.array([s, a[station], r, s1]), [1, 4])) #use a[nn] for action taken by that specific agent

            if total_steps > pre_train_steps and j>warmup_time:
                # start training here
                if e > endE:
                    e -= stepDrop
                #We train the selected agent
                if total_steps % (update_freq) == 0:
                    if total_steps % 150 ==0: #update target network every 80 seconds
                        t1=time.time()
                        stand_agent[nn].update_target_net()
                        t2=time.time()-t1
                    # Reset the recurrent layer's hidden state
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
                    trainBatch = stand_agent[nn].buffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    stand_agent[nn].train(trainBatch,trace_length,state_train,batch_size)


            # update reward
            rAll += r

            # swap state
            s = s1
            sP = s1P
            state = state1

        # Add the episode to the experience buffer
        for station in range(N_station):
            bufferArray = np.array(episodeBuffer[station])
            tempbuffer = list(zip(bufferArray))
            stand_agent[station].remember(tempbuffer)

        jList.append(j)
        rList.append(rAll)  # reward in this episode
        print('Episode:', i, ', totalreward:', rAll)


        # Periodically save the model.
        # if i % 100 == 0 and i != 0:
        #     saver.save(sess, path + '/model-' + str(i) + '.cptk')
        #     print("Saved Model")
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_steps, np.mean(rList[-summaryLength:]), e)
    #             saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
#                 summaryLength,h_size,sess,mainQN,time_per_step)
# saver.save(sess,path+'/model-'+str(i)+'.cptk')