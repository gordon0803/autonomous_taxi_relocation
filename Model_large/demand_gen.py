import numpy as np
import json
import math
import pickle
import taxi_util

simulation_input = dict()

N_station = 50;
distance = np.loadtxt(open('nycdata/selected_dist.csv','rb'),delimiter=',')
travel_time = np.loadtxt(open('nycdata/selected_time.csv','rb'),delimiter=',')
OD_mat=np.loadtxt(open('nycdata/od_50.csv','rb'),delimiter=',')


OD_mat=OD_mat/1400;  #every 20 seconds
travel_time=travel_time*15;
arrival_rate=OD_mat.sum(axis=1) #row sum for passenger arrival at the station
incoming__taxi=OD_mat.sum(axis=0)

OD_mat=OD_mat.tolist()



taxi_input = 50


simulation_input['N_station'] = N_station;
simulation_input['distance'] = distance
simulation_input['travel_time'] = travel_time
simulation_input['taxi_input'] = taxi_input
simulation_input['OD_mat'] = OD_mat
simulation_input['arrival_rate'] = arrival_rate

#relo_graph = taxi_util.RGraph(distance, incoming_taxi, arrival_rate)
# relo_graph = taxi_util.RGraph_reward(travel_time, np.array(OD_mat), arrival_rate)
# simulation_input['RG']=relo_graph

with open('simulation_input.dat', 'wb') as fp:
    pickle.dump(simulation_input, fp)
