import tensorflow as tf
import numpy as np
import random


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

class bandit_buffer():
        def __init__(self, buffer_size=5000):
            self.buffer = []
            self.buffer_size = buffer_size

        def add(self, experience):
            if len(self.buffer) + 1 >= self.buffer_size:
                self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
            self.buffer.append(experience)

        def sample(self, batch_size):
            sampled_episodes = []
            for i in range(batch_size):
                sampled_episodes.append(random.choice(self.buffer))
            sampledTraces = []
            for episode in sampled_episodes:
                sampledTraces.append(episode)
            sampledTraces = np.array(sampledTraces)
            return np.reshape(sampledTraces, [batch_size, 7])


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
    input_dim=4;
    return np.reshape(state,[Nstation*Nstation*input_dim])


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