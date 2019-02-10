
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random

class Qnetwork():
    def __init__(self, N_station, h_size, rnn_cell, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        # input is a scalar which will later be reshaped
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 3], dtype=tf.float32)

        # input is a tensor, like a 3 chanel image
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, N_station, N_station, 3])

        # create 4 convolution layers first
        self.conv1 = tf.layers.conv2d( \
            inputs=self.imageIn, filters=32, \
            kernel_size=[4, 4], strides=[3, 3], padding='VALID', \
             name=myScope + '_net_conv1')
        self.conv2 = tf.layers.conv2d( \
            inputs=self.conv1, filters=64, \
            kernel_size=[2, 2], strides=[2, 2], padding='VALID', \
             name=myScope + '_net_conv2')

        # self.conv4 = tf.nn.relu(tf.layers.conv2d( \
        #     inputs=self.conv3, filters=64, \
        #     kernel_size=[2, 2], strides=[2, 2], padding='VALID', \
        #      name=myScope + '_net_conv4'),name=myScope+'_net_relu4')
        self.trainLength = tf.placeholder(dtype=tf.int32,name=myScope+'_trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[],name=myScope+'_batchsize')
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.convFlat = tf.reshape(slim.flatten(self.conv2), [self.batch_size, self.trainLength, h_size],name=myScope+'_convlution_flattern')
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_net_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size],name=myScope+'_reshapeRNN_out')
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1,name=myScope+'_split_streamAV')
        self.AW = tf.Variable(tf.random_normal([h_size // 2, N_station]),name=myScope+'AW')
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]),name=myScope+'VW')
        self.Advantage = tf.matmul(self.streamA, self.AW, name=myScope+'_matmulAdvantage')
        self.Value = tf.matmul(self.streamV, self.VW, name=myScope+'_matmulValue')
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True), name=myScope+'_Qout')
        self.predict = tf.argmax(self.Qout, 1,name=myScope+'_prediction')
        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.actions_onehot = tf.one_hot(self.actions, N_station, dtype=tf.float32,name=myScope+'_onehot')
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])


        self.salience = tf.gradients(self.Advantage, self.imageIn)
        # Then combine them together to get our final Q-values.
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1, name=myScope+'Qvalue')
        self.td_error = tf.square(self.targetQ - self.Q, name=myScope+'_TDERROR')
        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.loss = tf.reduce_mean(self.td_error * self.mask, name=myScope+'_defineloss')
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001, name=myScope+'_Adam')
        self.updateModel = self.trainer.minimize(self.loss, name=myScope+'_training')



class experience_buffer():
    def __init__(self, buffer_size=30):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes=[]
        for i in range(batch_size):
            sampled_episodes.append(random.choice(self.buffer))
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 4])





#functions
def updateTarget(op_holder,sess):
    sess.run(op_holder)
    # total_vars = len(tf.trainable_variables())
    # a = tf.trainable_variables()[0].eval(session=sess)
    # b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    # if not a.all() == b.all():
    #     print("Target Set Failed")
    # else:
    #     print("Target set Success")


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value()))) #tau*main_network + (1-tau)*target network, soft update
    return op_holder


def processState(state,Nstation):
    #input is the N by N by 3 tuple, map it to a list
    return np.reshape(state,[Nstation*Nstation*3])


def compute_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()