#Zengxiang Lei 2019-02-13

#A tracker of system behavior, can do two things: 
#1. Record the simulation process into json file. 
#2. Load the simulation information and display it in a friendly way.

# Usage: 
# Five lines to enable the system_tracker
# 1. import it in the main.py 
# 2. Initialization before starting simulation 
# 3. Update_episode when new episode begin
# 4. Record info for every timestep
# 5. Save the log file as json
# For details please refer GREEDY_main.py and track the location of "sys_tracker"
# Every log file now should with size 58.1MB

import json
import numpy as np
from datetime import datetime
from pprint import pprint

class system_tracker():
	def __init__(self):
		self.baseinfo = dict()
		self.frameinfo=dict()
		self.episode_count =0
		self.timestep = 0

	def initialize(self, distance, travel_time, arrival_rate, taxi_input, N_station):
		self.baseinfo['distance'] = distance.tolist()
		self.baseinfo['travel_time'] = travel_time.tolist()
		self.baseinfo['arrival_rate'] = arrival_rate
		self.baseinfo['taxi_input'] = int(taxi_input) # transform numpy.int64 to normal integer
		self.N_station = N_station
		self.N_station_pair = N_station*N_station

	def new_episode(self):
		self.episode_count +=1
		self.frameinfo[str(self.episode_count)] = []

	def record(self, s, a):
		passenger_gap = np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
		taxi_in_travel = np.reshape(s[range(1,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station))
		taxi_in_relocation = np.reshape(s[range(2,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station))
		taxi_in_charge = np.diag(np.reshape(s[range(3,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
		action = [int(x) for x in a]
		oneframeinfo = {
		"passenger_gap":passenger_gap.tolist(), 
		#"taxi_in_travel": taxi_in_travel.tolist(), 
		#"taxi_in_relocation": taxi_in_relocation.tolist(),
		"taxi_in_charge":taxi_in_charge.tolist(), 
		"action": action
		}
		self.frameinfo[str(self.episode_count)].append(oneframeinfo)

	def save(self, name):
		data = dict()
		data['baseinfo'] = self.baseinfo
		data['frameinfo'] = self.frameinfo
		#print(data['frameinfo'])
		with open('log/sim_log_'+name+'_'+datetime.now().strftime('%Y-%m-%d %H-%M-%S')+'.json', 'w') as outfile:
		    json.dump(data, outfile)

	def load(self, filename):
		with open(filename) as f:
			data = json.load(f)
		self.baseinfo = data['baseinfo']
		self.frameinfo = data['frameinfo']
		pprint(self.baseinfo)

	def playback(self, episode, mode ='default'):
		if mode == 'verbose':
			pass # not done yet
		else:
			pprint(self.frameinfo[str(episode)])



