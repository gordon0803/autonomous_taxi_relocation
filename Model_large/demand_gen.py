import numpy as np
import json
import math
import pickle
import taxi_util
import glob

simulation_input = dict()

N_station = 40;
distance = np.loadtxt(open('nycdata/selected_dist.csv','rb'),delimiter=',')
travel_time = np.loadtxt(open('nycdata/selected_time.csv','rb'),delimiter=',')
distance=distance[:N_station,:N_station]
travel_time=travel_time[:N_station,:N_station]
#load the list of OD files, and normalize them to proper time interval
OD_mat=[]
normalize_od=120;
total_demand=0
for file in sorted(glob.glob('od_mat/*.csv'), key=lambda name: int(name[10:-4])):
    print(file)
    tempOD=np.genfromtxt(file, delimiter=',')
    tempOD/=normalize_od #convert into every 30 seconds
    OD_mat.append(tempOD[:N_station,:N_station])
    total_demand+=tempOD.sum()*normalize_od

#OD_mat=np.loadtxt(open('nycdata/od_70.csv','rb'),delimiter=',')

#convert arrival rate into 
travel_time=travel_time*4

#process arrival input, each item is the 48 time intervals for each station
arrival_rate=[[] for i in range(N_station)] #initialize
for i in range(len(OD_mat)):
    demand=OD_mat[i].sum(axis=1) #row sum
    for j in range(len(demand)):
        arrival_rate[j].append(demand[j])
        
OD_mat=[i.tolist() for i in OD_mat]

exp_dist=[[] for i in range(len(OD_mat))] #expected trip distance starting at each station for each time interval
                            
for t in range(len(OD_mat)):
    for i in range(N_station):
        v=0
        for j in range(N_station):
            if distance[i,j]==0 and OD_mat[t][i][j]>0:
                distance[i,j]=1

            v+=distance[i,j]*OD_mat[t][i][j]
        sumv=sum(OD_mat[t][i])
        if sumv>0:
            exp_dist[t].append(v/sum(OD_mat[t][i]))
        else:
            exp_dist[t].append(0)

print(len(exp_dist))


taxi_input = 50


simulation_input['N_station'] = N_station;
simulation_input['distance'] = distance
simulation_input['travel_time'] = travel_time
simulation_input['taxi_input'] = taxi_input
simulation_input['OD_mat'] = OD_mat
simulation_input['arrival_rate'] = arrival_rate
simulation_input['exp_dist']=exp_dist
                            
with open('simulation_input.dat', 'wb') as fp:
    pickle.dump(simulation_input, fp)
