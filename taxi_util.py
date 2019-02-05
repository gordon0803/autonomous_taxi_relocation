## Author: Xinwu Qian
## Last Update: 2019-2-5

#utility functions that will be used in the taxi_env


import math
import random
#update the waiting time of passengers in Q

def waiting_time_update(waiting_time):
	#input is the list of waiting time
	new_wait=[]

	#the list of leave probability based on the waiting time of passengers in the queue
	leave_prob=[math.exp(i/6.0)/(math.exp(i/6.0)+1000000) for i in waiting_time];

	for i in range(len(waiting_time)):
		if random.random()>leave_prob[i]:
			#the passenger will stay in the queue
			time=waiting_time[i]+1
			new_wait.append(time)

	return new_wait

