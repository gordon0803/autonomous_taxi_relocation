## Author: Xinwu Qian
## Last Update: 2019-2-5

#utility functions that will be used in the taxi_env

import tensorflow as tf
import math
import random
import numpy as np
from collections import deque
#update the waiting time of passengers in Q

def waiting_time_update(waiting_time,expect_waiting_time):
    (new_wait, new_expect_wait) = map(deque, zip(*[(i+1,j) for i,j in zip(waiting_time,expect_waiting_time) if i<j]))
    return new_wait,new_expect_wait


def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)
	total_vars = len(tf.trainable_variables())
	a = tf.trainable_variables()[0].eval(session=sess)
	b = tf.trainable_variables()[total_vars//2].eval(session=sess)
	if not a.all() == b.all():
		print("Target Set Failed")


def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder


def processState(state,Nstation):
	#input is the N by N by 3 tuple, map it to a list
	return np.reshape(state,[Nstation*Nstation*3])