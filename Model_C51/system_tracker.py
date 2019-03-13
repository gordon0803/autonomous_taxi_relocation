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

# Update 2019-03-12
# 1. Add record mode options: 'Endpoint', only record the start and the end episode
# 2. Revise the result display, now the last episode would be replay at the end of simulation

import json
import numpy as np
from datetime import datetime
from pprint import pprint

class system_tracker():
	def __init__(self):
		self.baseinfo = dict()
		self.frameinfo= list()
		self.episode_count =0
		self.timestep = 0
		self.recording = False

	def initialize(self, distance, travel_time, arrival_rate, taxi_input, N_station, num_episode, max_epLength,  mode = 'Endpoint'):
		self.baseinfo['distance'] = distance.tolist()
		self.baseinfo['travel_time'] = travel_time.tolist()
		self.baseinfo['arrival_rate'] = arrival_rate
		self.baseinfo['taxi_input'] = int(taxi_input) # transform numpy.int64 to normal integer
		self.baseinfo['N_station'] = N_station
		self.baseinfo['N_epilength'] = max_epLength
		self.N_station = N_station
		self.max_epLength = self.baseinfo['N_epilength']
		self.N_station_pair = N_station*N_station
		self.total_taxi = N_station*taxi_input
		print('total_taxi:', taxi_input)
		if(mode == 'Endpoint'):
			self.record_episode = [0, num_episode]
		else: 
			self.record_episode = list(range(num_episode))

	def new_episode(self):
		if (self.episode_count in self.record_episode):
			self.frameinfo.append(list())
			self.recording = True
		else:
			self.recording = False
		self.episode_count +=1

	def record(self, s, a):
		if(self.recording):
			passenger_gap = np.diag(np.reshape(s[range(0,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
			taxi_in_travel = np.reshape(s[range(1,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station))
			taxi_in_relocation = np.reshape(s[range(2,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station))
			taxi_in_charge = np.diag(np.reshape(s[range(3,len(s),len(s)//self.N_station_pair)],(self.N_station,self.N_station)))
			action = [int(x) for x in a]
			oneframeinfo = {
			"taxi_in_travel": taxi_in_travel.sum()*self.total_taxi,
			"taxi_in_relocation": taxi_in_relocation.sum()*self.total_taxi,
			"passenger_gap":(passenger_gap*self.total_taxi).tolist(),
			"taxi_in_charge":(taxi_in_charge*self.total_taxi).tolist(),
			"action": action
			}
			self.frameinfo[-1].append(oneframeinfo)

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
		self.N_station = self.baseinfo['N_station']
		self.max_epLength = self.baseinfo['N_epilength']
		pprint(self.baseinfo)

	def playback(self, episode, mode ='analytic'):
		print('-----------------------------------------------------------------')
		if mode == 'analytic':
			dataset = self.frameinfo[episode]
			avg, std, counts = self.process_data(dataset)
			print('Summary (per step):\t', 'average \t std')
			print('Taxi in travel    :\t %.3f \t %.2f' %  (avg[0], std[0]))
			print('Taxi in relocation:\t %.3f \t %.2f'%  (avg[1], std[1]))
			print('Passenger gap (Mean): ', ["%.2f" % v for v in avg[2]])
			print('Passenger gap (Std) : ', ["%.2f" % v for v in std[2]])
			print('Taxi in charge (Mean): ', ["%.2f" % v for v in avg[3]])
			print('Taxi in charge (Std) : ', ["%.2f" % v for v in std[3]])
			for i in range(len(counts)):

				print('Relocatoin choice ',i,': ', ["%s: %.2f" % (k,v) for k,v in counts[i].items()])
		else:
			pprint(self.frameinfo[episode])

	def process_data(self, dataset):
		# taxi in travel: average, std
		# taxi in relocation: average, std
		# passenger gap: average, std
		# taxi in charge: average, std
		# action: count 

		avg = [0,0,[0]*self.N_station,[0]*self.N_station]
		std = [0,0,[0]*self.N_station,[0]*self.N_station]
		actions = []
		counts= []
		for i in range(self.N_station):
			actions.append(list())
			counts.append(dict())
		for oneframeinfo in dataset:
			avg[0]+= oneframeinfo['taxi_in_travel']
			avg[1]+=oneframeinfo['taxi_in_relocation']
			std[0]+= oneframeinfo['taxi_in_travel']**2
			std[1]+=oneframeinfo['taxi_in_relocation']**2
			for i in range(self.N_station):
				avg[2][i]+=oneframeinfo['passenger_gap'][i]
				avg[3][i]+=oneframeinfo['taxi_in_charge'][i]
				std[2][i]+=oneframeinfo['passenger_gap'][i]**2
				std[3][i]+=oneframeinfo['taxi_in_charge'][i]**2
				actions[i].append(oneframeinfo['action'][i])
		avg[0] = avg[0]/self.max_epLength
		avg[1] = avg[1]/self.max_epLength
		std[0] = np.sqrt(std[0]/self.max_epLength-avg[0]**2)
		std[1] = np.sqrt(std[1]/self.max_epLength-avg[1]**2)
		for i in range(self.N_station):
			avg[2][i] = avg[2][i]/self.max_epLength
			avg[3][i] = avg[3][i]/self.max_epLength
			std[2][i] = np.sqrt(std[2][i]/self.max_epLength-avg[2][i]**2)
			std[3][i] = np.sqrt(std[3][i]/self.max_epLength-avg[3][i]**2)
			unique, count = np.unique(actions[i], return_counts=True)
			for key, value in zip(unique, count):
				counts[i][str(key)] = value/self.max_epLength
		return avg, std, counts






