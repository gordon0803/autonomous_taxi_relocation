#Xinwu Qian 2019-02-06

#Main file for DRQN


import os,sys
sys.path.insert(0,'./config')
import taxi_env as te
import DRQN_agent
import network
import config
import pickle
from datetime import datetime
import tensorflow as tf
import numpy as np

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

reward_out=open('log/reward_log_'+datetime.now().strftime('%Y-%m-%d %H-%M-%S')+'.csv', 'w+')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open('simulation_input.dat','rb') as fp:
    simulation_input=pickle.load(fp)

#------------------Parameter setting-----------------------
N_station=simulation_input['N_station']
OD_mat=simulation_input['OD_mat']
distance=simulation_input['distance']
travel_time=simulation_input['travel_time']
arrival_rate=simulation_input['arrival_rate']
taxi_input=simulation_input['taxi_input']
print(arrival_rate)


#Setting the training parameters
batch_size = config.TRAIN_CONFIG['batch_size']
trace_length = config.TRAIN_CONFIG['trace_length'] #How long each experience trace will be when training
update_freq = config.TRAIN_CONFIG['update_freq'] #How often to perform a training step.
y = config.TRAIN_CONFIG['y'] #Discount factor on the target Q-values
startE =config.TRAIN_CONFIG['startE'] #Starting chance of random action
endE = config.TRAIN_CONFIG['endE'] #Final chance of random action
anneling_steps =config.TRAIN_CONFIG['anneling_steps'] #How many steps of training to reduce startE to endE.
num_episodes = config.TRAIN_CONFIG['num_episodes'] #How many episodes of game environment to train network with.
load_model = config.TRAIN_CONFIG['load_model'] #Whether to load a saved model.
warmup_time=config.TRAIN_CONFIG['warmup_time'];
path = "./drqn" #The path to save our model to.
h_size = config.TRAIN_CONFIG['h_size']
max_epLength = config.TRAIN_CONFIG['max_epLength']
pre_train_steps = max_epLength*50 #How many steps of random actions before training begins.
softmax_action=config.TRAIN_CONFIG['softmax_action']

tau = 0.001


#--------------Simulation initialization
env=te.taxi_simulator(arrival_rate,OD_mat,distance,travel_time,taxi_input)
env.reset()
print('System Successfully Initialized!')
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

    global_init=tf.global_variables_initializer()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    # writer.close()
    sess.run(global_init)



    for i in range(num_episodes):
        episodeBuffer = [[] for station in range(N_station)]

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
        # We train one station in one single episode, and hold it unchanged for other stations, and we keep rotating.
        nn = i % N_station
        print(nn)

        # Reset the recurrent layer's hidden state
        state = [(np.zeros([1, h_size]), np.zeros([1, h_size])) for station in range(N_station)]
        state1=[[] for station in range(N_station)]
        # The Q-Network
        while j < max_epLength:

            j += 1
            # for all the stations, act greedily
            # Choose an action by greedily (with e chance of random action) from the Q-network

            a=[-1]*N_station

            if softmax_action==True:  #use softmax
                for station in range(N_station):
                    state1[station] = stand_agent[station].get_rnn_state(s, state[station])
                    if env.taxi_in_q[station]:
                        Qdist = stand_agent[station].predict_softmax(s, state)
                        Qprob=network.compute_softmax(Qdist);
                        a1_v=np.random.choice(Qprob[0],p=Qprob[0])
                        a1=np.argmax(Qprob[0] == a1_v)
                    else:
                        a1=station

                    a[station] = a1  # action performed by DRQN

            else: #use e-greedy
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            state1[station] = stand_agent[station].get_rnn_state(s, state[station])
                            a[station]=np.random.randint(0, N_station) #random actions for each station
                        else:
                            a[station]=station
                else:
                    for station in range(N_station):
                        if env.taxi_in_q[station]:
                            state1[station] = stand_agent[station].get_rnn_state(s, state[station])
                            a1 = stand_agent[station].predict(s,state[station])
                        else:
                            a1=station #self-relocation

                        a[station]=a1[0] #action performed by DRQN


            # move to the next step based on action selected
            ssp, lfp = env.step(a)
            total_serve+=ssp
            total_leave+=lfp

            # get state and reward
            s1P, r,rp = env.get_state()

            s1 = network.processState(s1P, N_station)

            total_steps += 1

            # episode buffer
            # we don't store the initial 200 steps of the simulation, as warm up periods
            if j>warmup_time:
                #we penalize the reward to motivate long term benefits
                # newr=r+rp[a[nn]]  #system reward + shared reward
                newr = r
                #only record the buffer for the chosen agent
                for station in range(N_station):
                    episodeBuffer[station].append(np.reshape(np.array([s, a[station], newr, s1]), [1, 4])) #use a[nn] for action taken by that specific agent

            if total_steps > pre_train_steps and j>warmup_time:
                # start training here
                if e > endE:
                    e -= stepDrop
                #We train the selected agent
                if total_steps % (update_freq) == 0:
                    stand_agent[nn].update_target_net() #soft update target network

                    # Reset the recurrent layer's hidden state
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
                    trainBatch = stand_agent[nn].buffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    stand_agent[nn].train(trainBatch,trace_length,state_train,batch_size)



            # update reward after the warm up period
            if j>warmup_time:
                rAll += r

            # swap state
            s = s1
            sP = s1P
            for station in range(N_station):
                state[station] = state1[station]

        # Add the episode to the experience buffer
        for station in range(N_station):
            bufferArray = np.array(episodeBuffer[station])
            tempArray=[]
            #now we break this bufferArray into tiny steps, according to the step length
            for point in range(len(bufferArray) + 1 - trace_length):
                tempArray.append(bufferArray[point:point + trace_length])
            stand_agent[station].remember(tempArray)

        jList.append(j)
        rList.append(rAll)  # reward in this episode
        print('Episode:', i, ', totalreward:', rAll, ', total serve:',total_serve,', total leave:',total_leave)


        reward_out.write(str(i)+','+str(rAll)+'\n')


        #Periodically save the model.
#         if i % 100 == 0 and i != 0:
#             saver.save(sess, path + '/model-' + str(i) + '.cptk')
#             print("Saved Model")
#         if len(rList) % summaryLength == 0 and len(rList) != 0:
#             print(total_steps, np.mean(rList[-summaryLength:]), e)
#                 saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
#                 summaryLength,h_size,sess,mainQN,time_per_step)

# saver.save(sess,path+'/model-'+str(i)+'.cptk')
reward_out.close()
