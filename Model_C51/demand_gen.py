import numpy as np
import json
import math
import pickle
import config
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
#arrival_rate = [(i + 1) / 6.0 for i in range(N_station)]
#arrival_rate=[0.1,0.4,0.7,1,1.3,0.1,0.4,0.7,1,1.3]


#temporally varying passenger demand
arrival_rate=[[0.1,0.3,0.2,0.15],[0.4,0.6,0.8,0.4],[0.7,1.1,1.3,1.1],[1,1.3,0.9,1.3],[1.3,1.6,1.3,1.5],[0.1,0.3,0.2,0.15],[0.4,0.6,0.8,0.4],[0.7,1.1,1.3,1.1],[1,1.3,0.9,1.3],[1.3,1.6,1.3,1.5]]
#parse arriva rate into N_stationX timesteps list with random number generaterd before hands
rng_seed=config.TRAIN_CONFIG['random_seed'];


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

taxi_input = 10

simulation_input['N_station'] = N_station;
simulation_input['distance'] = distance
simulation_input['travel_time'] = travel_time
simulation_input['taxi_input'] = taxi_input
simulation_input['OD_mat'] = OD_mat
simulation_input['arrival_rate'] = arrival_rate
simulation_input['exp_dist']=exp_dist



with open('simulation_input.dat', 'wb') as fp:
    pickle.dump(simulation_input, fp)
