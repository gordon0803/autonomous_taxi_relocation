import numpy as np
import json
import math
import pickle

simulation_input=dict()

N_station=10;
l1=[5 for i in range(N_station)]
OD_mat=[l1 for i in range(N_station)]
distance=np.zeros((N_station,N_station))
for i in range(N_station):
        distance[i,i] = 0;
        for j in range(N_station):
            distance[i,j]=math.ceil(abs(j-i)/2);

travel_time=distance
arrival_rate=[(i*2+1)/6.0 for i in range(N_station)]
taxi_input=10

simulation_input['N_station']=N_station;
simulation_input['distance']=distance
simulation_input['travel_time']=travel_time
simulation_input['taxi_input']=taxi_input
simulation_input['OD_mat']=OD_mat
simulation_input['arrival_rate']=arrival_rate

with open('simulation_input.dat','wb') as fp:
	pickle.dump(simulation_input,fp)