##Configuration file here


TRAIN_CONFIG = {
    'batch_size':48,
    'trace_length': 10,
    'update_freq': 30,
    'lstm_unit':256,
    'y': .99,
    'elimination_threshold':.5,
    'startE':1,
    'endE':0.05,
    'anneling_steps':200*1000,
    'num_episodes':250,
    'buffer_size':5000,
    'prioritized':0,
    'load_model':False,
    'warmup_time':-1,
    'model_path':'./drqn',
    'h_size':800, #The size of the final convolutional layer before splitting it into Advantage and Value streams.
    'max_epLength':1000, #The max allowed length of our episode.
    'pre_train_steps':20000, #How many steps of random actions before traning begins
    'softmax_action':False, #use softmax or not
    'silent': 1, #0 for print, 1 for no print
    'use_linear':1,
    'use_tracker':0,
    'random_seed':0 #specify the random seed used across the experiments
}


NET_CONFIG={
    'eta':0.8  #how to alter the reward function
}

#No experience replay, masking first 10 elementswf
