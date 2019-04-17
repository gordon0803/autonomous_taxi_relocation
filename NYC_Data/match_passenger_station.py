#Match taxi passengers with parking garages in Manhattan

from math import radians, cos, sin, asin, sqrt
import csv
import time
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def distance(coord1,coord2):
    return(haversine(coord1[0],coord1[1],coord2[0],coord2[1]))

#load parking data
parking_coord=[];
with open('parking_lot_manhattan.txt','r') as f:
    next(f)
    for lines in f:
        line=lines.strip().split('\t')
        lat=float(line[8])
        lon=float(line[9])
        parking_coord.append([lon,lat])
nstation=len(parking_coord)

nselected = 100

station_flow=[[0]*nstation for i in range(nstation)]
selected_flow = [[0]*nselected for i in range(nselected)]
#load passenger data and do the matching
total_satisfied=0;
total_demand=0.0;

# First run the passenger matching for one time
with open('processed_NYC_0909.csv','r') as f:
    for lines in f:
        line = lines.strip().split(',')
        if(time.strptime(line[0],'%Y-%m-%d %H:%M:%S').tm_hour>=7 and time.strptime(line[0],'%Y-%m-%d %H:%M:%S').tm_hour<9):
            ocoord=[float(line[2]),float(line[3])]
            dcoord=[float(line[4]),float(line[5])]
            o_id = -1
            d_id = -1
            min_d_o_to_o=1e10
            min_d_d_to_d=1e10
            for i in range(len(parking_coord)):
                d_o_to_o= distance(ocoord,parking_coord[i]) #distance from origin to the station
                if d_o_to_o<min_d_o_to_o:
                    min_d_o_to_o=d_o_to_o
                    o_id=i
            for j in range(len(parking_coord)):
                d_d_to_d=distance(dcoord,parking_coord[j])
                if d_d_to_d<min_d_d_to_d:
                    min_d_d_to_d = d_d_to_d
                    d_id=j
            if min_d_o_to_o<0.3 and min_d_d_to_d<0.3: 
                station_flow[o_id][d_id]+=1;

# Pick the first [nselected] station sorted decreasingly by demand
demand = np.sum(station_flow, axis = 0)
selected_stations = demand.argsort()[-nselected:][::-1]
print(selected_stations)
selected_coord = [parking_coord[k] for k in  selected_stations]
with open('processed_NYC_0909.csv','r') as f:
    for lines in f:
        line = lines.strip().split(',')
        if(time.strptime(line[0],'%Y-%m-%d %H:%M:%S').tm_hour>=7 and time.strptime(line[0],'%Y-%m-%d %H:%M:%S').tm_hour<9):
            ocoord=[float(line[2]),float(line[3])]
            dcoord=[float(line[4]),float(line[5])]
            o_id = -1
            d_id = -1
            min_d_o_to_o=1e10
            min_d_d_to_d=1e10
            for i in range(len(selected_coord)):
                d_o_to_o= distance(ocoord,selected_coord[i]) #distance from origin to the station
                if d_o_to_o<min_d_o_to_o:
                    min_d_o_to_o=d_o_to_o
                    o_id=i
            for j in range(len(selected_coord)):
                d_d_to_d=distance(dcoord,selected_coord[j])
                if d_d_to_d<min_d_d_to_d:
                    min_d_d_to_d = d_d_to_d
                    d_id=j
            if min_d_o_to_o<0.3 and min_d_d_to_d<0.3: 
                selected_flow[o_id][d_id]+=1;
                total_satisfied+=1
            total_demand+=1

    

print(total_satisfied)
print(total_satisfied/total_demand)

with open("ouput_700_7_9.csv",'wb') as f:
    writer=csv.writer(f)
    writer.writerows(selected_flow)

travel_dist = np.loadtxt('travel_dist.csv', delimiter=",")
travel_dist = travel_dist[selected_stations,:][:,selected_stations]
np.savetxt("selected_dist.csv", travel_dist, delimiter=",")
travel_time = np.loadtxt('travel_time.csv', delimiter=",")
travel_time = travel_time[selected_stations,:][:,selected_stations]
np.savetxt("selected_time.csv", travel_time, delimiter=",")
print("Complete")
