#test if the taxi_environment is coded properly

import taxi_env
import time
from collections import deque
import numpy as np


N_station = 100;
distance = np.loadtxt(open('nycdata/selected_dist.csv','rb'),delimiter=',')
travel_time = np.loadtxt(open('nycdata/selected_time.csv','rb'),delimiter=',')
OD_mat=np.loadtxt(open('nycdata/od_100.csv','rb'),delimiter=',')


OD_mat=OD_mat/180;  #every 20 seconds
travel_time=travel_time*3;
arrival_rate=OD_mat.sum(axis=0) #row sum for passenger arrival at the station
incoming__taxi=OD_mat.sum(axis=1)

OD_mat=OD_mat.tolist()


taxi_input = 50



taxi_simulator=taxi_env.taxi_simulator(arrival_rate,OD_mat,distance,travel_time,taxi_input)
taxi_simulator.reset()
print('System Successfully Initialized!')

pass_gap,taxi_in_travel,taxi_in_relocation,reward=taxi_simulator.env_summary()

print('System reward:',reward)
print('Passenger state:',pass_gap)
print('Travel state:',taxi_in_travel)
print('Relocation state:',taxi_in_relocation)

start=time.time()
for i in range(1000):
    print(i)
    taxi_simulator.step([-1 for i in range(N_station)])
    # taxi_state,reward=taxi_simulator.get_state()

end=time.time()-start
print('Time per step:',end/1000)