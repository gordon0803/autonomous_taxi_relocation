#test if the taxi_environment is coded properly

import taxi_env
import time
from collections import deque
N_station=10
l1=[5 for i in range(N_station)]
OD_mat=[l1 for i in range(N_station)]

distance=OD_mat

travel_time=OD_mat

arrival_rate=[1 for i in range(N_station)]

taxi_input=10


taxi_simulator=taxi_env.taxi_simulator(arrival_rate,OD_mat,distance,travel_time,taxi_input)
taxi_simulator.reset()
print('System Successfully Initialized!')

pass_gap,taxi_in_travel,taxi_in_relocation,reward=taxi_simulator.env_summary()

print('System reward:',reward)
print('Passenger state:',pass_gap)
print('Travel state:',taxi_in_travel)
print('Relocation state:',taxi_in_relocation)

start=time.time()
act=[3 for i in range(N_station)]
for i in range(1000):
    taxi_simulator.step(act)
    taxi_state,reward,a,b=taxi_simulator.get_state()


end=time.time()-start
print('Time per step:',end/1000, 'total time:', end)