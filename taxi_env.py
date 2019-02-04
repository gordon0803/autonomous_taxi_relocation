## Author: Xinwu Qian
## Last Update: 2019-2-4

## This is the code for the taxi environment for use in DRQN
## Agent is the stand, action is the place to relocate, and state is a tensor of passenger state, occupie vehicles, and relocate vehicles

import numpy as np
import random as rd

class taxi_agent():

	#each taxi serves as an agent,where we need to track the origin, destination, and the battery of each taxis
	def __init__(self, battery_miles): #provide the capacity of the battery
		self.battery=200 #remaining battery
		self.max_battery=200 #battery capacity
		self.idle=True
		self.destination=None
		self.time_to_destination=0;

	#taxi assigned to a passenger
	def trip(self,destination,time_to_destination,distance):
		self.idle=False;
		self.destination=destination;
		self.time_to_destination=time_to_destination;
		self.batter+=-distance; #deduct the distance for the battery of the vehicle

	#taxi now moves
	def move(self): 
		self.time_to_destination+=-1; #getting close to the destination

	# if the taxi arrived at the destination, reset its parameters
	def arrived(self):
		self.idle=True;
		self.destination=None;
		self.time_to_destination=0;




class taxi_simulator():
	def __init__(self, lambda, OD_mat, dist_mat, time_mat):
		#number of taxi stands
		self.N=len(lambda)
		#arrival rate
		self.arrival_rate=lambda
		#departure rate of all taxi stands
		self.departure_rate=lambda
		#OD ratio matrix of all taxi stands
		self.OD_split=OD_mat
		#travel distance
		self.distance=dist_mat
		#travel time of each stands
		self.travle_time=time_mat



		#Following are the states of the game
		self.passenger_state=np.zeros((N,1)) #number of passenegrs in Q at each station
		self.occupied_taxi=np.zeros((N,N))  #number of occupied taxis in trip
		self.relocate_taxi=np.zeros((N,N)) #number of taxis during relocation
		self.passenger_qtime=[[] for i in range(N)]  #each station has a list for counting waiting time of each passengers
		self.taxi_in_travel=[];
		self.taxi_in_relocation=[];
		self.taxi_in_q=[[] for i in range(N)] #taxis waiting in the queue of each station
		self.taxi_in_charge=[[] for i in range(N)] #taxis charging at each station

	#1: all taxi traveling
	def taxi_travel():
		for taxi in self.taxi_in_travel:
			taxi.move

	#2: all taxi charging
	def taxi_charging():
		#loop through all taxis in the charging dock, if they finished charging, send them back to q
		#one time step would charge 0.5% of the battery
		for i in range(self.N):
			for j in range(len(self.taxi_in_charge[i])):
				taxi=self.taxi_in_charge[i].pop()
				taxi.battery+=0.005*taxi.max_battery
				if taxi.battery>=taxi.max_battery:
					taxi.battery=taxi.max_battery
					self.taxi_in_q[i].append(taxi)
				else:
					self.taxi_in_charge[i].append(taxi)


	#3: determine if the taxi arrive at destination
	def taxi_arrive():
		ntaxi=len(self.taxi_in_travel)
		for i in range(len(ntaxi)):
			taxi=self.taxi_in_travel.pop() 
			if taxi.time_to_destination<=0: 
				#arrived
				if taxi.battery>=0.2*taxi.max_battery:
					self.taxi_in_q[taxi.destination].append(taxi) #add this taxi to the stands at destination
				else:
					self.taxi_in_charge[taxi.destination].append(taxi)
				
				taxi.arrived()
			else:
				#still in travel, send this taxi back to the list
				self.taxi_in_travel.append(taxi) 




	#generate passengers
	def pass_gen(self): 
		for i in raneg(self.N):
			n_pass_arrive=np.random.poisson(self.arrival_rate[i])
			#add passenegrs to the queue
			for j in range(n_pass_arive):
				self.qtime[i].append(0)


