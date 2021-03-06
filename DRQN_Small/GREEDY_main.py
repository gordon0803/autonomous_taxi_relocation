#Zengxiang Lei 2019-02-09

#Main file for greedy relocation


import os
from network import *
import taxi_env as te
import GREEDY_agent
import network
import time
import math
import config

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

reward_out=open('log/reward_log_greedy.csv', 'w')  #Replace the old log

#------------------Parameter setting-----------------------
N_station=10
l1=[5 for i in range(N_station)]
OD_mat=[l1 for i in range(N_station)]
distance=np.zeros((N_station,N_station))
for i in range(N_station):
        distance[i][i] = 0;
        for j in range(N_station):
            distance[i][j]=math.ceil(abs(j-i)/2);

travel_time=distance
print(travel_time)
arrival_rate=[0.5+i/3.0 for i in range(N_station)]
taxi_input=10


env=te.taxi_simulator(arrival_rate,OD_mat,distance,travel_time,taxi_input)
env.reset()
print('System Successfully Initialized!')

#Setting the training parameters
num_episodes = config.TRAIN_CONFIG['num_episodes'] #How many episodes of game environment to train network with.
warmup_time=config.TRAIN_CONFIG['warmup_time'];
max_epLength = config.TRAIN_CONFIG['max_epLength']
pre_train_steps = max_epLength*50 #How many steps of random actions before training begins.
softmax_action=config.TRAIN_CONFIG['softmax_action']


#------------------Train the network-----------------------


# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0


# # this step loads the model from the model that has been saved
# if load_model == True:
#     print('Loading Model...')
#     ckpt = tf.train.get_checkpoint_state(path)
#     saver.restore(sess, ckpt.model_checkpoint_path)

# this example equals target network to the original network after every few episodes
# we may want to modify this

stand_agent = []
# targetOps=[]

for station in range(N_station):
	stand_agent.append(GREEDY_agent.greedy_agent(str(station), N_station))


for i in range(num_episodes):
    episodeBuffer = []

    # Reset environment and get first new observation
    env.reset()
    # return the current state of the system
    sP, tempr,temprp = env.get_state()
    # process the state into a list
    s = network.processState(sP, N_station)

    rAll = 0
    j = 0
    total_serve = 0
    total_leave = 0

    # The Greedy movement
    while j < max_epLength:

       j += 1
       a=[-1]*N_station
       # for all the stations, act greedily
       # Choose an action by greedily (with gap) for this time
       if softmax_action==True:  #use softmax
           for station in range(N_station):
               prob=stand_agent[station].predict_softmax(s)
               a1=np.random.choice(list(range(N_station)),1,p=prob)[0]
               a[station] = a1  # action performed by DRQN

       else: #use max gap
           for station in range(N_station):
               a1 = stand_agent[station].predict(s)
               a[station]=a1 #action performed by DRQN

       # move to the next step based on action selected
       ssp, lfp = env.step(a)
       total_serve+=ssp
       total_leave+=lfp

       # get state and reward
       s1P, r,rp = env.get_state()

       s1 = network.processState(s1P, N_station)

       total_steps += 1

       # update reward after the warm up period
       if j>warmup_time:
           rAll += r

       # swap state
       s = s1
       sP = s1P


    jList.append(j)
    rList.append(rAll)  # reward in this episode
    print('Episode:', i, ', totalreward:', rAll, ', total serve:',total_serve,', total leave:',total_leave)

reward_out.write(str(j)+','+str(rAll)+'\n')
reward_out.close()
