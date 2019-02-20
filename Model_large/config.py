##Configuration file here
import demand_gen
import pickle

TRAIN_CONFIG = {
    'batch_size':10,
    'trace_length': 20,
    'update_freq': 200,
    'y': .99,
    'startE':1,
    'endE':0.05,
    'anneling_steps':600*1000,
    'num_episodes':1000,
    'buffer_size':5000,
    'prioritized':0,
    'load_model':False,
    'warmup_time':200,
    'model_path':'./drqn',
    'h_size':64, #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    'max_epLength':1000, #The max allowed length of our episode.
    'pre_train_steps':20000, #How many steps of random actions before traning begins
    'softmax_action':False, #use softmax or not
    'silent': 0, #0 for print, 1 for no print
    'use_RG':1, #use relocation graph or not
    'use_tracker':0 #use system tracker or not
}


NET_CONFIG={


}

#No experience replay, masking first 10 elements