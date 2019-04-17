##Configuration file here
import demand_gen
import pickle

TRAIN_CONFIG = {
    'batch_size':32,
    'trace_length': 10,
    'update_freq': 30,
    'lstm_unit':256,
    'y': .99,
    'elimination_threshold':.8,
    'startE':1,
    'endE':0.05,
    'anneling_steps':200*1500,
    'num_episodes':500,
    'buffer_size':5000,
    'prioritized':0,
    'load_model':False,
    'warmup_time':-1,
    'model_path':'./drqn',
    'h_size':800, #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    'max_epLength':1500, #The max allowed length of our episode.
    'pre_train_steps':20000, #How many steps of random actions before traning begins
    'softmax_action':False, #use softmax or not
    'silent': 1, #0 for print, 1 for no print
    'use_linear':1,
    'use_tracker':1
}


NET_CONFIG={


}

#No experience replay, masking first 10 elementswf
