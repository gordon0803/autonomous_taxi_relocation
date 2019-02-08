## Author: Xinwu Qian
## Last Update: 2019-2-5

## This is the code for the taxi environment for use in DRQN
## Agent is the stand, action is the place to relocate, and state is a tensor of passenger state, occupied vehicles, and relocated vehicles

import numpy as np
import random as rd
import taxi_util as util
from collections import deque
from numba import jit


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

        # number of taxi stands
        self.N = len(arrival_rate)
        # list of station IDs from 0 to N-1
        self.station_list = np.array([i for i in range(self.N)])
        # passenger arrival rate
        self.arrival_rate = arrival_rate
        # OD ratio matrix of all taxi stands
        self.OD_split=OD_mat

        # travel distance
        self.distance = dist_mat
        # travel time of each stands
        self.travel_time = time_mat
        # taxi input
        self.taxi_input = taxi_input

        self.passenger_qtime = [deque([])  for i in range(self.N)]  # each station has a list for counting waiting time of each passengers
        self.passenger_expect_wait = [deque([]) for i in range(self.N)]  # expected wait time of each passenger
        self.passenger_destination=[deque([])  for i in range(self.N)] #destination of the passenger
        self.taxi_in_travel = deque([])
        self.taxi_in_relocation = deque([])
        self.taxi_in_q = [deque([])  for i in range(self.N)]  # taxis waiting in the queue of each station
        self.taxi_in_charge = [deque([])  for i in range(self.N)]  # taxis charging at each station
        self.gamma_pool=np.random.gamma(10,size=500000).tolist() #maintain a pool of gamma variable of size 50000

    # assign the initial list of taxis
    def init_taxi(self):
        # taxi_input can be a scalar or a vector
        # if scalar: each station has k taxis
        taxi_input = self.taxi_input
        if not isinstance(taxi_input, list):
            taxi_input = [taxi_input for i in range(self.N)]  # convert it to a list

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
            self.gamma_pool = np.random.gamma(10, size=500000).tolist()

        for i in range(len(action)):
            if action[i] > -1:
                if self.taxi_in_q[i]:
                    taxi = self.taxi_in_q[i].popleft()  # relocate the first taxi
                    time_to_destination = self.travel_time[i][action[i]]  # from i to action[i]
                    distance_to_destination = self.distance[i][action[i]]
                    taxi.trip(i, action[i], time_to_destination, distance_to_destination)
                    self.taxi_in_relocation.append(taxi)
                # else:
                # 	print('No taxis in Q, check if your relocation action is derived properly')

        # now start the simulation

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



    # 2: all taxi traveling
    def taxi_travel(self):
        for taxi in self.taxi_in_travel:
            taxi.move()

    # 3: all taxi charging
    def taxi_charging(self):
        # loop through all taxis in the charging dock, if they finished charging, send them back to q
        # one time step would charge 0.5% of the battery
        for i in range(self.N):
            for j in range(len(self.taxi_in_charge[i])):
                if self.taxi_in_charge[i]:
                    taxi = self.taxi_in_charge[i].popleft()
                    taxi.battery += 0.005 * taxi.max_battery  # every time step, 0.5% of battery get charged
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

    # 5: update passenger waiting, leave, and generate passengers
    def passenger_update(self):
        for i in range(self.N):
            # update waiting time of existing passengers
            # also determine if the passengers will leave
            if self.passenger_qtime[i]:
                self.passenger_qtime[i], self.passenger_expect_wait[i] = util.waiting_time_update(self.passenger_qtime[i],self.passenger_expect_wait[i])
            # new passengers
            n_pass_arrive = np.random.poisson(self.arrival_rate[i])
            destination = np.random.choice(self.station_list, n_pass_arrive, self.OD_split[i]).tolist()
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
                    self.passenger_qtime[i].popleft()  # remove the first passenger
                    self.passenger_expect_wait[i].popleft()
                    if self.taxi_in_q[i]:
                        taxi = self.taxi_in_q[i].popleft()  # remove the first taxi in q
                        # first determine if the destination of the passenger
                        destination=self.passenger_destination[i].popleft()
                        time_to_destination = self.travel_time[i][destination]
                        distance_to_destination = self.distance[i][destination]
                        taxi.trip(i, destination, time_to_destination, distance_to_destination)
                        # send this taxi to travel Q
                        self.taxi_in_travel.append(taxi)
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
        self.gamma_pool = np.random.gamma(10, size=500000).tolist()  # maintain a pool of gamma variable of size 50000

    def get_state(self):
        # give the states of the system after all the executions
        # the state of the system is a 3 N by N matrix
        state = np.ones([self.N, self.N, 3])
        passenger_gap = np.zeros((self.N, self.N))
        taxi_in_travel = np.zeros((self.N, self.N))
        taxi_in_relocation = np.zeros((self.N, self.N))

        for i in range(self.N):
            passenger_gap[i, i] = len(self.passenger_qtime[i])

        for t in self.taxi_in_travel:
            taxi_in_travel[t.origin, t.destination] += 1

        for t in self.taxi_in_relocation:
            taxi_in_relocation[t.origin, t.destination] += 1

        state[:, :, 0] = passenger_gap;
        state[:, :, 1] = taxi_in_travel;
        state[:, :, 2] = taxi_in_relocation;
        # reward
        total_taxi_in_travel = taxi_in_travel.sum()
        total_taxi_in_relocation = taxi_in_relocation.sum()
        reward = total_taxi_in_travel - total_taxi_in_relocation

        return state, reward
