#Zengxiang Lei 2019-02-09

#Agent file for agents who follow greedy strategies

import os
import numpy as np
import tensorflow as tf
import network
import time

class greedy_agent():
    def __init__(self,name,N_station):
        #config is the parameter setting
        #ckpt_path is the path for load models
        self.name=name
        self.N_station = N_station
        self.N_station_pair = N_station*N_station

    def predict(self,s):
        #make the prediction
        passenger_gap = np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
        action=np.argmax(passenger_gap)
        return action

    def predict_softmax(self,s):
        passenger_gap = np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
        utility = np.exp(np.array(passenger_gap))
        prob = utility / sum(utility)
        return prob