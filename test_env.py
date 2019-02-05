#test if the taxi_environment is coded properly

import taxi_env


OD_mat=[[2,2,2],[2,2,2],[2,2,2]]

distance=[[2,2,2],[2,2,2],[2,2,2]]

time=[[2,2,2],[2,2,2],[2,2,2]]

arrival_rate=[5,10,7]

taxi_input=[5]


taxi_simulator=taxi_env.taxi_simulator(arrival_rate,OD_mat,distance,time)
print('System Successfully Initialized!')

pass_gap,taxi_in_travel,taxi_in_relocation,reward=taxi_simulator.env_summary()

print('System reward:',reward)
print('Passenger state:',pass_gap)
print('Travel state:',taxi_in_travel)
print('Relocation state:',taxi_in_relocation)







taxi_simulator.step([1,1,1])
pass_gap,taxi_in_travel,taxi_in_relocation,reward=taxi_simulator.env_summary()

print('System reward:',reward)
print('Passenger state:',pass_gap)
print('Travel state:',taxi_in_travel)
print('Relocation state:',taxi_in_relocation)