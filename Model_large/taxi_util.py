## Author: Xinwu Qian
## Last Update: 2019-2-5

#utility functions that will be used in the taxi_env
from collections import deque
#update the waiting time of passengers in Q

import networkx as nx
import numpy as np


def waiting_time_update(waiting_time,expect_waiting_time):
    try:
        (new_wait, new_expect_wait) = map(deque, zip(*[(i+1,j) for i,j in zip(waiting_time,expect_waiting_time) if i<j]))
    except ValueError:
        new_wait=deque([])
        new_expect_wait=deque([])

    return new_wait,new_expect_wait



#Giving the distance matrix, the arrival rate, and departure rate of each station,
#Returning the adjacency matrix that has minimum maximum degree with lowest weight, that balancing the unbalance demand of the network
def RGraph(dist,arrive,depart):
    #dist: N by N distance matrix
    #arrive: N by 1 total arrival rate
    #depart: N by 1 total departure rate
    gap=np.array(arrive)-np.array(depart)
    dist=np.array(dist)

    N=len(arrive) #number of stations

    dist = dist #introduce some noise to avoid two edges with the same weight being selected

    G=np.ones((N,N)) #starting with a complete graph

    #first we create a bipartite graph:

    for i in range(N):
        for j in range(N):
            if gap[i] * gap[j] > 0: #same sign for the gap, do not connect them
                if i != j:
                    G[i, j] = 0
                    G[j, i] = 0
                    dist[i,j]=1e10 #very large number so no one will use it
                    dist[j,i]=1e10
    #stands with positive and negative weights
    pos=[]
    neg=[]
    for i in range(N):
        if gap[i]>=0:
            pos.append(i)
        else:
            neg.append(i)

    G1=nx.DiGraph()
    source=N;
    sink=N+1;
    #construct G1 from G, source, and sink: sink connects to all nodes in neg, source connects to all nodes in pos
    for i in range(N):
        for j in range(N):
            if i in pos and j in neg:
                G1.add_edge(i,j,weight=dist[i,j])
                print(dist[i,j])
                # G1.add_edge(j,i,weight=dist[j,i],capacity=1e10)
    #link source to pos:
    for i in range(N):
        if i in pos:
            G1.add_edge(source,i,weight=0,capacity=abs(gap[i]))
        elif i in neg:
            G1.add_edge(i,sink,weight=0,capacity=abs(gap[i]))

    mincostflow=nx.max_flow_min_cost(G1,source,sink)
    # mincost=nx.cost_of_flow(G1,mincostflow)

    #now construct the graph
    newG=np.eye(N) #diagonal matrix
    for i in range(N):
        for j in range(N):
            if j in mincostflow[i].keys():
                if mincostflow[i][j]>0:
                    newG[i,j]=1
                    newG[j,i]=1


    N_edge_complete=N**2;
    N_edge_cp2=G.sum();
    N_edge_new=newG.sum();
    print("Number of edge reduced by:", N_edge_complete-N_edge_new)
    print("Number of edge reduced by:", N_edge_cp2 - N_edge_new)
    return newG

