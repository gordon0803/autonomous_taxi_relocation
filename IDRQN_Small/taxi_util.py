## Author: Xinwu Qian
## Last Update: 2019-2-5

#utility functions that will be used in the taxi_env
from collections import deque
#update the waiting time of passengers in Q

def waiting_time_update(waiting_time,expect_waiting_time):
    try:
        (new_wait, new_expect_wait) = map(deque, zip(*[(i+1,j) for i,j in zip(waiting_time,expect_waiting_time) if i<j]))
    except ValueError:
        new_wait=deque([])
        new_expect_wait=deque([])

    return new_wait,new_expect_wait


