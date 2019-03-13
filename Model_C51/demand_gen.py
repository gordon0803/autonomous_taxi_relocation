import numpy as np
import json
import math
import pickle
import taxi_util
import time

simulation_input = dict()

N_station = 10;
l1 = [5 for i in range(N_station)]
distance = np.zeros((N_station, N_station))
for i in range(N_station):
    distance[i, i] = 0;
    for j in range(N_station):
        distance[i, j] = 2 * math.ceil(abs(j - i) / 2);

travel_time = distance
arrival_rate = [(i + 1) / 6.0 for i in range(N_station)]
arrival_rate=[0.1,0.4,0.7,1,1.3,0.1,0.4,0.7,1,1.3]

OD_mat = []
for i in range(N_station):
    kk = [(i * 2 + 1) / 6.0 for i in range(N_station)]
    kk[i] = 0
    OD_mat.append(kk)
print(OD_mat)

exp_dist=[] #expected trip distance starting at each station
for i in range(N_station):
    v=0
    for j in range(N_station):
        v+=distance[i,j]*OD_mat[i][j]

    exp_dist.append(v/sum(OD_mat[i]))

print(exp_dist)
# calculate taxi_arrival_rate at each station
incoming_taxi = np.zeros(N_station)
for i in range(N_station):
    temp_in = 0;
    for j in range(N_station):
        rate = np.array(OD_mat[j]) / sum(OD_mat[j])
        temp_in += arrival_rate[j] * rate[i]
    incoming_taxi[i] = temp_in

taxi_input = 6

simulation_input['N_station'] = N_station;
simulation_input['distance'] = distance
simulation_input['travel_time'] = travel_time
simulation_input['taxi_input'] = taxi_input
simulation_input['OD_mat'] = OD_mat
simulation_input['arrival_rate'] = arrival_rate
simulation_input['exp_dist']=exp_dist

t1=time.time()
#relo_graph = taxi_util.RGraph(distance, incoming_taxi, arrival_rate)
print('generate network in:',time.time()-t1)


with open('simulation_input.dat', 'wb') as fp:
    pickle.dump(simulation_input, fp)
