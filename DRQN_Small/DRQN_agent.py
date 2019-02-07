#Xinwu Qian 2019-02-07

#Agent file for agents who follow DRQN to update their rewards

import os
import numpy as np
import tensorflow as tf
import network


class drqn_agent():
    def __init__(self,name,N_station,h_size,tau,sess,ckpt_path=None):
        #config is the parameter setting
        #ckpt_path is the path for load models
        self.name=name
        self.buffer=network.experience_buffer() #each agent holds its own experience replay buffer
        self.action=-1 #remember the most recent action taken
        self.ckpt_path=ckpt_path
        self.sess=sess;
        self.drqn_build(N_station,h_size,tau) #build the network



    def drqn_build(self,N_station,h_size,tau):
        #construct the DRQN for each agent

        self.cell = tf.contrib.rnn.LSTMBlockCell(num_units=h_size,name='main_lstm'+self.name)
        self.cellT = tf.contrib.rnn.LSTMBlockCell(num_units=h_size,name='target_lstm'+self.name)

        #build main and target network
        self.mainQN = network.Qnetwork(N_station, h_size, self.cell, 'main_network_'+self.name)
        self.targetQN = network.Qnetwork(N_station, h_size, self.cell, 'target_network_' + self.name)


        #saver
        self.saver=tf.train.Saver()


        #load model from path
        if self.ckpt_path:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print("Cannot restore model, does not exist")
                raise Exception
        else:
            self.sess.run(tf.global_variables_initializer())

        self.trainables=tf.trainable_variables() #get the set of variables and then send to update target network
        #store the name and initial values for target network
        # self.targetOps=network.updateTargetGraph(self.trainables,tau)
        self.update_target_net()

        print("Agent network initialization complete, Agent name:",self.name)

    def update_target_net(self):
        #set target network's parameters to be the same as the primary network
        """
        Copies the model parameters of one estimator to another.
        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith('main_network_'+self.name)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith('target_network_'+self.name)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)

        # network.updateTarget(self.targetOps,self.sess)

    def predict(self,s,state):
        #make the prediction
        action=self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput:[s],self.mainQN.trainLength:1,\
                                                      self.mainQN.state_in:state,self.mainQN.batch_size:1})

        return action

    def get_rnn_state(self,s,state):
        state1=self.sess.run(self.mainQN.rnn_state, feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1, \
                                                              self.mainQN.state_in: state, self.mainQN.batch_size: 1})

        return state1


    def train(self,trainBatch,trace_length,state_train,batch_size):
        #use double DQN as the training step
        #Use main net to make a prediction
        Q1=self.sess.run(self.mainQN.predict, feed_dict={ \
            self.mainQN.scalarInput: np.vstack(trainBatch[:, 3]),self.mainQN.trainLength: trace_length,\
            self.mainQN.state_in: state_train, self.mainQN.batch_size: batch_size})
        #Use target network to evaluate output
        Q2 = self.sess.run(self.targetQN.Qout, feed_dict={ \
            self.targetQN.scalarInput: np.vstack(trainBatch[:, 3]), \
            self.targetQN.trainLength: trace_length, self.targetQN.state_in: state_train, self.targetQN.batch_size: batch_size})

        #Metl the Q value to obtain the target Q value
        doubleQ = Q2[range(batch_size * trace_length), Q1]
        targetQ = trainBatch[:, 2] + (.99 * doubleQ) #.99 is the discount for doubleQ value

        #update network parameters with the predefined training method
        self.sess.run(self.mainQN.updateModel, \
                 feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ, \
                            self.mainQN.actions: trainBatch[:, 1], self.mainQN.trainLength: trace_length, \
                            self.mainQN.state_in: state_train, self.mainQN.batch_size: batch_size})


    #remember the episodebuffer
    def remember(self,episodeBuffer):
        self.buffer.add(episodeBuffer)



