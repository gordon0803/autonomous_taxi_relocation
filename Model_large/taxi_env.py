## Author: Xinwu Qian
## Last Update: 2019-2-5

## This is the code for the taxi environment for use in DRQN
## Agent is the stand, action is the place to relocate, and state is a tensor of passenger state, occupied vehicles, and relocated vehicles

import numpy as np
import taxi_util as util
from collections import deque
import math



class taxi_agent():
    # each taxi serves as an agent,where we need to track the origin, destination, and the battery of each taxis
    def __init__(self, battery_miles):  # provide the capacity of the battery
        self.battery = battery_miles  # remaining battery
        self.max_battery = battery_miles  # battery capacity
        self.idle = True
        self.destination = None
        self.time_to_destination = 0

    # taxi assigned to a passenger
    def trip(self, origin, destination, time_to_destination, distance):
        self.idle = False
        self.origin = origin
        self.destination = destination
        self.time_to_destination = time_to_destination
        self.battery -= distance  # deduct the distance for the battery of the vehicle

    # taxi now moves
    def move(self):
        self.time_to_destination += -1  # getting close to the destination

    # if the taxi arrived at the destination, reset its parameters
    def arrived(self):
        self.idle = True
        self.destination = None
        self.time_to_destination = 0

class taxi_simulator():
    def __init__(self, arrival_rate, OD_mat, dist_mat, time_mat, taxi_input):
        # lamba is a vector of size 1 by N
        # OD_mat,dist_mat, and time_mat are list of list of size N by N
        self.timer=0 #current step
        # number of taxi stands
        self.N = len(arrival_rate)
        # list of station IDs from 0 to N-1
        self.station_list = np.array([i for i in range(self.N)])
        # passenger arrival rate
        self.arrival_input = arrival_rate

        # OD ratio matrix of all taxi stands
        self.OD_split=OD_mat

        # travel distance
        self.distance = dist_mat
        # travel time of each stands
        self.travel_time = time_mat
        # taxi input
        self.taxi_input = taxi_input

        #current action
        self.action=[0]*self.N #no action made


        self.passenger_qtime = [deque([])  for i in range(self.N)]  # each station has a list for counting waiting time of each passengers
        self.passenger_expect_wait = [deque([]) for i in range(self.N)]  # expected wait time of each passenger
        self.passenger_destination=[deque([])  for i in range(self.N)] #destination of the passenger
        self.taxi_in_travel = deque([])
        self.taxi_in_relocation = deque([])
        self.taxi_in_q = [deque([])  for i in range(self.N)]  # taxis waiting in the queue of each station
        self.taxi_in_charge = [deque([])  for i in range(self.N)]  # taxis charging at each station
        # self.gamma_pool=np.random.gamma(10,size=500000).tolist() #maintain a pool of gamma variable of size 50000
        self.gamma_pool=(10*np.ones(500000)).tolist()

        self.served_passengers = np.zeros(self.N)
        self.served_passengers_waiting_time = np.zeros(self.N)
        self.leaved_passengers = np.zeros(self.N)
        self.leaved_passengers_waiting_time = np.zeros(self.N)

    # assign the initial list of taxis
    def init_taxi(self):
        # taxi_input can be a scalar or a vector
        # if scalar: each station has k taxis
        taxi_input = self.taxi_input
        #create random taxi initial numbers
        rnd_array = np.random.multinomial(taxi_input*self.N, np.ones(self.N) / self.N, size=1)[0]
        if not isinstance(taxi_input, list):
            taxi_input = rnd_array

        self.total_taxi=sum(taxi_input)

        for i in range(self.N):
            for t in range(taxi_input[i]):
                taxi = taxi_agent(200)  # battery is set to 200 here
                self.taxi_in_q[i].append(taxi)

    # 1: execute actions
    def step(self, action):
        # input is the relocation action each station made
        # --------------CAUTION--------------
        # Actions are made outside of the simulation environment
        # A relocation action is made if there are available taxis to move
        # We move up to 1 taxi at a time
        # Use -1 to denote no relocation
        # check gamma pool
        if len(self.gamma_pool) < 10000:
            self.gamma_pool = (10*np.ones(500000)).tolist()
        
        self.current_action=[-1]*self.N
        self.clock=self.timer//120 #decide which time interval it falls into

        for i in range(len(action)):
            if action[i] > -1:
                self.current_action[i]=action[i] #remember the action taken
                if self.taxi_in_q[i]:
                    taxi = self.taxi_in_q[i].popleft()  # relocate the first taxi
                    time_to_destination = self.travel_time[i][action[i]]  # from i to action[i]
                    distance_to_destination = self.distance[i][action[i]]
                    taxi.trip(i, action[i], time_to_destination, distance_to_destination)
                    self.taxi_in_relocation.append(taxi)
                # else:
                # 	print('No taxis in Q, check if your relocation action is derived properly')

        # now start the simulation
        self.served_pass=0;
        self.left_pass=0;
        # step 1: travel
        self.taxi_travel()
        # step 2: charging
        self.taxi_charging()
        # step 3: arrive
        self.taxi_arrive()
        # step 4: update passenger
        self.passenger_update()
        # step 5: serve passengers
        self.passenger_serve()

        self.timer+=1

        return self.served_pass,self.left_pass

    # 2: all taxi traveling
    def taxi_travel(self):
        for taxi in self.taxi_in_travel:
            taxi.move()
        for taxi in self.taxi_in_relocation:
            taxi.move()

    # 3: all taxi charging
    def taxi_charging(self):
        # loop through all taxis in the charging dock, if they finished charging, send them back to q
        # one time step would charge 0.5% of the battery
        for i in range(self.N):
            for j in range(len(self.taxi_in_charge[i])):
                if self.taxi_in_charge[i]:
                    taxi = self.taxi_in_charge[i].popleft()
                    taxi.battery += 0.1 * taxi.max_battery  # every time step, 0.5% of battery get charged
                    if taxi.battery >= taxi.max_battery:
                        taxi.battery = taxi.max_battery
                        self.taxi_in_q[i].append(taxi)
                    else:
                        # not fully charged, send it back to the charging q
                        self.taxi_in_charge[i].append(taxi)

    # 4: determine if the taxi arrive at destination
    def taxi_arrive(self):
        ntaxi = len(self.taxi_in_travel)
        for i in range(ntaxi):
            if self.taxi_in_travel:
                taxi = self.taxi_in_travel.popleft()
                if taxi.time_to_destination <= 0:
                    # arrived
                    if taxi.battery >= 0.2 * taxi.max_battery:
                        self.taxi_in_q[taxi.destination].append(taxi)  # add this taxi to the stands at destination
                    else:
                        self.taxi_in_charge[taxi.destination].append(taxi)

                    taxi.arrived()
                else:
                    # still in travel, send this taxi back to the travel list
                    self.taxi_in_travel.append(taxi)

        ntaxi = len(self.taxi_in_relocation)
        for i in range(ntaxi):
            if self.taxi_in_relocation:
                taxi = self.taxi_in_relocation.popleft()
                if taxi.time_to_destination <= 0:
                    # arrived
                    if taxi.battery >= 0.2 * taxi.max_battery:
                        self.taxi_in_q[taxi.destination].append(taxi)  # add this taxi to the stands at destination
                    else:
                        self.taxi_in_charge[taxi.destination].append(taxi)

                    taxi.arrived()
                else:
                 # still in travel, send this taxi back to the travel list
                    self.taxi_in_relocation.append(taxi)
    # 5: update passenger waiting, leave, and generate passengers
    def passenger_update(self):
        for i in range(self.N):
            # update waiting time of existing passengers
            # also determine if the passengers will leave

            tp=len(self.passenger_qtime[i])
            if self.passenger_qtime[i]:
                self.passenger_qtime[i], self.passenger_expect_wait[i], left_waiting_time = util.waiting_time_update(self.passenger_qtime[i],self.passenger_expect_wait[i])

                self.left_pass+=len(self.passenger_qtime[i])-tp
                self.leaved_passengers[i]+=tp-len(self.passenger_qtime[i])
                self.leaved_passengers_waiting_time[i]+=left_waiting_time

            # new passengers
            n_pass_arrive = self.arrival_rate[i][self.timer]
            destination = np.random.choice(self.station_list, n_pass_arrive, self.OD_split[self.clock][i]).tolist()
            # add passengers to the queue
            # new passengers with 0 waiting time
            #a different way of getting ride of for loop
            self.passenger_qtime[i]+=[0]*n_pass_arrive
            expect_wait_append=[self.gamma_pool.pop() for j in range(n_pass_arrive)]
            self.passenger_expect_wait[i]+=expect_wait_append
            # del self.gamma_pool[:n_pass_arrive]
            self.passenger_destination[i] += destination


            # for j in range(n_pass_arrive):
            #     self.passenger_qtime[i].append(0)
            #     self.passenger_expect_wait[i].append(self.gamma_pool.pop())  # expected wait length = 10
            #     self.passenger_destination[i].append(destination[j])



    # 6: serve all the passengers with the available taxis
    def passenger_serve(self):
        for i in range(self.N):
            # loop over all stands
            for j in range(len(self.taxi_in_q[i])):
                # loop over available taxis in the stand
                # serve passenger
                if len(self.passenger_qtime[i]) > 0:
                    waiting_time = self.passenger_qtime[i].popleft()  # remove the first passenger
                    self.passenger_expect_wait[i].popleft()
                    #if self.taxi_in_q[i]:
                    taxi = self.taxi_in_q[i].popleft()  # remove the first taxi in q
                    # first determine if the destination of the passenger
                    destination=self.passenger_destination[i].popleft()
                    time_to_destination = self.travel_time[i][destination]
                    distance_to_destination = self.distance[i][destination]
                    taxi.trip(i, destination, time_to_destination, distance_to_destination)
                    # send this taxi to travel Q
                    self.taxi_in_travel.append(taxi)
                    self.served_pass+=1
                    # record information
                    self.served_passengers[i]+=1
                    self.served_passengers_waiting_time[i]+=waiting_time
                else:
                    # no available passengers, break the loop
                    break

    # 7: system states summary
    def env_summary(self):
        # give the states of the system after all the executions
        # the state of the system is a 3 N by N matrix
        passenger_gap = np.zeros((self.N, self.N))
        taxi_in_travel = np.zeros((self.N, self.N))
        taxi_in_relocation = np.zeros((self.N, self.N))

        # reward
        total_taxi_in_travel = len(self.taxi_in_travel)
        total_taxi_in_relocation = len(self.taxi_in_relocation)
        reward = total_taxi_in_travel - total_taxi_in_relocation

        for i in range(self.N):
            passenger_gap[i, i] = len(self.passenger_qtime[i])

        for t in self.taxi_in_travel:
            taxi_in_travel[t.origin, t.destination] += 1

        for t in self.taxi_in_relocation:
            taxi_in_relocation[t.origin, t.destination] += 1

        return passenger_gap, taxi_in_travel, taxi_in_relocation, reward

    def reset(self):
        # reset the state to initial
        self.passenger_qtime = [deque([])  for i in range(self.N)]  # each station has a list for counting waiting time of each passengers
        self.passenger_expect_wait = [deque([]) for i in range(self.N)]  # expected wait time of each passenger
        self.passenger_destination=[deque([])  for i in range(self.N)] #destination of the passenger
        self.taxi_in_travel = deque([])
        self.taxi_in_relocation = deque([])
        self.taxi_in_q = [deque([])  for i in range(self.N)]  # taxis waiting in the queue of each station
        self.taxi_in_charge = [deque([])  for i in range(self.N)]  # taxis charging at each station
        self.init_taxi()
        self.gamma_pool = (10*np.ones(500000)).tolist()  # maintain a pool of gamma variable of size 50000
        #current action
        self.previous_action=[-1]*self.N #no action made
        self.current_action=[-1]*self.N

        #probability of getting a passenger at a particular station
        self.score=[1]*self.N
        self.served_passengers = np.zeros(self.N)
        self.served_passengers_waiting_time = np.zeros(self.N)
        self.leaved_passengers = np.zeros(self.N)
        self.leaved_passengers_waiting_time = np.zeros(self.N)

        self.timer=0;

        #pregeneration demand
        self.arrival_rate = []
        max_time_step = 2880;
        steps = max_time_step // (len(self.arrival_input[0]) ) -1
        x_base = [steps * i for i in range(len(self.arrival_input[0]))]
        x_project = [i for i in range(max_time_step)]
        for i in range(self.N):
            arrive = np.interp(x_project, x_base, self.arrival_input[i])
            arrive = np.random.poisson(arrive).tolist()
            self.arrival_rate.append(arrive)



    def get_state(self):
        # give the states of the system after all the executions
        # the state of the system is a 3 N by N matrix
        max_passenger=50;
        state = np.ones([self.N, self.N, 5])
        passenger_gap = np.zeros((self.N, self.N))
        taxi_in_travel = np.zeros((self.N, self.N))
        taxi_in_relocation = np.zeros((self.N, self.N))
        taxi_in_charge=np.zeros((self.N,self.N))
        taxi_in_q=np.zeros((self.N,self.N))
        previous_action=np.zeros((self.N,self.N))

        incoming_taxi=np.array([0]*self.N)
        awaiting_pass=np.array([0]*self.N)

        for i in range(self.N):
            if self.taxi_in_charge[i]:
                taxi_in_charge[i,i]=len(self.taxi_in_charge[i])
            if self.taxi_in_q[i]:
                taxi_in_q[i,i]=len(self.taxi_in_q[i])
            if not self.previous_action[i]==-1:
                previous_action[i,self.previous_action[i]]=1


        self.previous_action=self.current_action #swap


        for i in range(self.N):
            passenger_gap[i, i] = min(len(self.passenger_qtime[i]),max_passenger)/max_passenger
            awaiting_pass[i]=len(self.passenger_qtime[i])
        #
        for t in self.taxi_in_travel:
            taxi_in_travel[t.origin, t.destination] += 1
            incoming_taxi[t.destination]+=1

        for t in self.taxi_in_relocation:
            if not t.origin==t.destination:  #self relocation will not count, viewed as stay
                taxi_in_relocation[t.origin, t.destination] += 1
                incoming_taxi[t.destination]+=1

        #normalize for taxis
        taxi_in_travel=taxi_in_travel/self.total_taxi;
        taxi_in_relocation=taxi_in_relocation/self.total_taxi;
        taxi_in_charge=taxi_in_charge/self.total_taxi;
        taxi_in_q=taxi_in_q/self.total_taxi;


        #all states are within 0-1, continuous value

        state[:, :, 0] = passenger_gap;
        state[:, :, 1] = taxi_in_travel;
        state[:, :, 2] = taxi_in_relocation;
        state[:, :, 3] = taxi_in_q;
        state[:,:,4] = taxi_in_charge;
        # reward
        total_taxi_in_travel = taxi_in_travel.sum()
        total_taxi_in_relocation = taxi_in_relocation.sum()
        total_taxi_stay=taxi_in_q.sum()
        reward = -sum(awaiting_pass)/max_passenger-total_taxi_in_relocation
        oldreward=total_taxi_in_travel-total_taxi_in_relocation


        #calculate linear features and scores
        feature=[]
        score=[]
        for i in range(self.N):
            feature+=[passenger_gap[i,i],taxi_in_q[i,i],taxi_in_relocation[:,i].sum(),taxi_in_travel[:,i].sum()]
            #update score
            if self.taxi_in_q[i]: #drivers waiting passengers
                # self.score[i]*=sigmoid(-min(len(self.taxi_in_q[i]),20))
                # self.score[i]=max(self.score[i],0.1)
                self.score[i]=0
            else:
                # self.score[i]*=sigmoid(min(len(self.passenger_qtime[i]),20))
                # self.score[i]=min(self.score[i],1) #bound to [0,1]
                self.score[i]=len(self.passenger_qtime[i])

            score.append(self.score[i])


        return state, reward, np.array(feature),np.array(score),reward


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def sigmoid(x):
    return 0.9+0.5/(1+math.exp(-x))
