##Configuration file here
import demand_gen
import pickle

TRAIN_CONFIG = {
    'batch_size': 10,
    'trace_length': 100,
    'update_freq': 5,
    'y': .99,
    'startE':1,
    'endE':0.01,
    'anneling_steps':300*500,
    'num_episodes':1000,
    'load_model':False,
    'warmup_time':-1,
    'model_path':'./drqn',
    'h_size':64, #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    'max_epLength':500, #The max allowed length of our episode.
    'pre_train_steps':20000, #How many steps of random actions before traning begins
    'softmax_action':False, #use softmax or not
    'silent': 0 #0 for print, 1 for no print
}


NET_CONFIG={


}
