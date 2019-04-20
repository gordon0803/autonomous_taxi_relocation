import numpy as np
import json
import math
import pickle
import taxi_util
import glob

simulation_input = dict()

N_station = 70;
distance = np.loadtxt(open('nycdata/selected_dist.csv','rb'),delimiter=',')
travel_time = np.loadtxt(open('nycdata/selected_time.csv','rb'),delimiter=',')

#load the list of OD files, and normalize them to proper time interval
OD_mat=[]
normalize_od=30;
for file in glob.glob('od_mat/*.csv'):
    print(file)
    tempOD=np.genfromtxt(file, delimiter=',')
    tempOD/=mormalized_od #convert into every minutes 
    OD_mat.append()
    
#OD_mat=np.loadtxt(open('nycdata/od_70.csv','rb'),delimiter=',')

travel_time=travel_time;

#convert arrival rate into 
travel_time=travel_time;

#process arrival input, each item is the 48 time intervals for each station
arrival_rate=[[] for i in range(N_station)] #initialize
for i in range(len(OD_mat)):
    demand=OD_mat.sum(axis=1) #row sum
    for j in range(len(demand)):
        arrival_rate[j].append(demand[j])

arrival_rate=OD_mat.sum(axis=1) #row sum for passenger arrival at the station
incoming__taxi=OD_mat.sum(axis=0)

OD_mat=OD_mat.tolist()

exp_dist=[] #expected trip distance starting at each station
for i in range(N_station):
    v=0
    for j in range(N_station):
        v+=distance[i,j]*OD_mat[i][j]

    exp_dist.append(v/sum(OD_mat[i]))

taxi_input = 40


simulation_input['N_station'] = N_station;
simulation_input['distance'] = distance
simulation_input['travel_time'] = travel_time
simulation_input['taxi_input'] = taxi_input
simulation_input['OD_mat'] = OD_mat
simulation_input['arrival_rate'] = arrival_rate
simulation_input['exp_dist']=exp_dist

#relo_graph = taxi_util.RGraph(distance, incoming_taxi, arrival_rate)
# relo_graph = taxi_util.RGraph_reward(travel_time, np.array(OD_mat), arrival_rate)
# simulation_input['RG']=relo_graph

with open('simulation_input.dat', 'wb') as fp:
    pickle.dump(simulation_input, fp)
