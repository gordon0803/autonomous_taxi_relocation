import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random

class linear_model():
    def __init__(self,N_station):
        #define the linear model to distinguish among actions

        self.W=tf.Variable(tf.constant(1/N_station,shape=[N_station*4,N_station]),name='linear_params')
        self.linear_X=tf.placeholder(shape=[None,N_station*4],dtype=tf.float32,name='linear_params_X')
        self.linear_X_reshape = tf.reshape(self.linear_X, shape=[-1, N_station*4])
        self.linear_Y = tf.placeholder(shape=[None, N_station], dtype=tf.float32,name='linear_params_Y')

        self.l1_regularizer = tf.contrib.layers.l2_regularizer(
            scale=0.01, scope=None
        )
        self.weights = tf.trainable_variables(scope='linear_params')  # all vars of your graph
        self.regularization_penalty = tf.contrib.layers.apply_regularization(self.l1_regularizer, self.weights)

        self.linear_Yh=tf.matmul(self.linear_X_reshape,self.W) #linear model
        self.linear_loss=tf.reduce_mean(tf.square(self.linear_Yh-self.linear_Y))+self.regularization_penalty
        self.linear_opt=tf.train.AdamOptimizer(learning_rate=0.1, name='linear_adam')
        self.linear_update=self.linear_opt.minimize(self.linear_loss,name='linear_train')



class Qnetwork():
    def __init__(self, N_station, h_size, batch_size,train_length, myScope,  is_gpu=0,prioritized=0):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.

        # input is a scalar which will later be reshaped
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 6], dtype=tf.float32)

        # input is a tensor, like a 3 chanel image
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, N_station, N_station, 6])

        # create 4 convolution layers first
        self.conv1 = tf.nn.relu(tf.layers.conv2d( \
            inputs=self.imageIn, filters=32, \
            kernel_size=[4, 4], strides=[3, 3], padding='VALID', \
             name=myScope + '_net_conv1'))
        self.conv2 = tf.nn.relu(tf.layers.conv2d( \
            inputs=self.conv1, filters=64, \
            kernel_size=[2, 2], strides=[2, 2], padding='VALID', \
             name=myScope + '_net_conv2'))



        # self.conv4 = tf.nn.relu(tf.layers.conv2d( \
        #     inputs=self.conv3, filters=64, \
        #     kernel_size=[3, 3], strides=[2, 2], padding='VALID', \
        #      name=myScope + '_net_conv4'),name=myScope+'_net_relu4')

        self.trainLength = tf.placeholder(dtype=tf.int32,name=myScope+'_trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[],name=myScope+'_batchsize')
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.convFlat = tf.reshape(slim.flatten(self.conv2), [self.batch_size, self.trainLength, h_size],name=myScope+'_convlution_flattern')

        if is_gpu:
            print('Using CudnnLSTM')
            self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=h_size,name=myScope+'_lstm')
            self.rnn, self.rnn_state = self.lstm(inputs=self.convFlat)
        else:
            print('Using LSTMfused')
            self.lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=h_size,name=myScope+'_lstm')
            self.rnn, self.rnn_state = self.lstm(inputs=self.convFlat, dtype=tf.float32)

        if prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [batch_size,train_length], name=myScope+'IS_weights')
            self.ISWeights_new=tf.reshape(self.ISWeights,[-1])



        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size],name=myScope+'_reshapeRNN_out')
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1,name=myScope+'_split_streamAV')
        self.AW = tf.Variable(tf.random_normal([h_size // 2, N_station+1]),name=myScope+'AW') #action +1, with the last action being station without any vehicles
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]),name=myScope+'VW')
        self.Advantage = tf.matmul(self.streamA, self.AW, name=myScope+'_matmulAdvantage')
        self.Value = tf.matmul(self.streamV, self.VW, name=myScope+'_matmulValue')
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True), name=myScope+'_Qout')
        self.predict = tf.argmax(self.Qout, 1,name=myScope+'_prediction')
        self.maskA = tf.zeros([self.batch_size, train_length//2]) #Mask first 20 records are shown to have the best results
        self.maskB = tf.ones([self.batch_size, train_length//2])
        self.actions_onehot = tf.one_hot(self.actions, N_station+1, dtype=tf.float32,name=myScope+'_onehot') #action +1, with the last action being station without any vehicles
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])


        self.salience = tf.gradients(self.Advantage, self.imageIn)
        # Then combine them together to get our final Q-values.
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1, name=myScope+'Qvalue')
        self.td_error = tf.square(self.targetQ - self.Q, name=myScope+'_TDERROR')
        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        if prioritized:
            self.abs_errors = tf.reduce_mean(tf.reshape(tf.abs(self.targetQ - self.Q),shape=[self.batch_size,self.trainLength]),axis=1)
            self.loss = tf.reduce_mean(self.td_error*self.mask,name=myScope+'_per_defineloss')
        else:
            self.loss = tf.reduce_mean(self.td_error * self.mask, name=myScope+'_defineloss')

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001, name=myScope+'_Adam')
        self.updateModel = self.trainer.minimize(self.loss, name=myScope+'_training')



class experience_buffer():
    def __init__(self, buffer_size=5000):
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
            sampledTraces.append(episode)
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 7])


#prioritized experience buffer
class per_experience_buffer():
    epsilon = 0.00001  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.6/(800*100)
    abs_err_upper = 1.  # clipped abs error


    def __init__(self, capacity=3000):
        self.tree = SumTree(capacity)
        self.capacity=capacity

    def add(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, batch_size,trace_length):
        b_idx, b_memory, ISWeights = np.empty((batch_size,), dtype=np.int32), [], np.empty((batch_size, trace_length))
        pri_seg = self.tree.total_p / batch_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        templist=self.tree.tree[-self.tree.capacity:]
        min_prob = np.min(templist[np.nonzero(templist)]) / self.tree.total_p

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, :] = np.power(prob/min_prob, -self.beta)
            b_idx[i]=idx
            b_memory.append(data)

        b_memory=np.reshape(b_memory, [batch_size * trace_length, 4])
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)




class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root



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
    #input is the N by N by 6 tuple, map it to a list
    return np.reshape(state,[Nstation*Nstation*6])


def compute_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def condition(ctr,stand_agent,batch_size,trace_length):
    return ctr < len(stand_agent)


def train_batch(ctr,stand_agent,batch_size,trace_length):
    #avoid overhead from tensorflow
    trainBatch = stand_agent[ctr].buffer.sample(batch_size,trace_length)  # Get a random batch of experiences.

# Below we perform the Double-DQN update to the target Q-values
    stand_agent[ctr].train(trainBatch, trace_length, batch_size)
    print('station:'+str(ctr)+' has been trained')
    return ctr+1,stand_agent,batch_size,trace_length