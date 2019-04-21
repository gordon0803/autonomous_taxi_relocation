#Zengxiang Lei 2019-02-09

#Agent file for agents who follow greedy strategies

import os
import numpy as np
import tensorflow as tf
import network
import time

class greedy_agent():
    def __init__(self,name,N_station,neighbor_loc,total_taxi):
        #config is the parameter setting
        #ckpt_path is the path for load models
        self.name=name
        self.N_station = N_station
        self.N_station_pair = N_station*N_station
        self.threshold =np.zeros(N_station)
        self.neighbor_loc = neighbor_loc[1:]
        self.total_taxi = total_taxi
        self.max_passenger =50
        self.time_step = 0


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

    def predict_inventory(self,s):
        inventory_gap = np.diag(np.reshape(s[range(3,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))*self.total_taxi-\
                        np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))*self.max_passenger
        action = -1
        #print(inventory_gap)
        if (inventory_gap[int(self.name)])>=self.threshold[int(self.name)]:
            # to the closest shortage place
            for neighbor_loc in self.neighbor_loc:
                if inventory_gap[neighbor_loc]<self.threshold[neighbor_loc]:
                    action = neighbor_loc
                    break

        # update estimation of expect inventory
        self.threshold = (self.threshold*self.time_step+np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))*self.max_passenger)/(self.time_step+1)
        self.time_step +=1
        return action
