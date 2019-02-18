#Xinwu Qian 2019-02-07

#Agent file for agents who follow DRQN to update their rewards

import os
import numpy as np
import tensorflow as tf
import network
import time

class drqn_agent():
    def __init__(self,name,N_station,h_size,tau,sess, batch_size,train_length,prioritized=0,is_gpu=0,ckpt_path=None):
        #config is the parameter setting
        #ckpt_path is the path for load models
        self.name=name
        if prioritized:
            self.buffer=network.per_experience_buffer() #prioritized experience replay buffer
            print('Use PER Buffer')
        else:
            self.buffer=network.experience_buffer()
            print('Use Normal Buffer')

        self.action=-1 #remember the most recent action taken
        self.ckpt_path=ckpt_path
        self.sess=sess;
        self.drqn_build(N_station,h_size,tau,batch_size,train_length,prioritized=prioritized,is_gpu=is_gpu) #build the network



    def drqn_build(self,N_station,h_size,tau,batch_size,train_length,prioritized,is_gpu):

        #build main and target network
        self.mainQN = network.Qnetwork(N_station, h_size, batch_size,train_length,myScope='Graph_'+self.name+'_main_network_'+self.name,is_gpu=is_gpu,prioritized=prioritized)
        self.targetQN = network.Qnetwork(N_station, h_size, batch_size,train_length,myScope='Graph_'+self.name+'_target_network_' + self.name, is_gpu=is_gpu,prioritized=prioritized)


        #saver
        # self.saver=tf.train.Saver()

        #load model from path
        # if self.ckpt_path:
        #     ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #     else:
        #         print("Cannot restore model, does not exist")
        #         raise Exception
        # else:
        #     self.sess.run(tf.global_variables_initializer())

        self.main_trainables=tf.trainable_variables(scope='Graph_'+self.name+'_main_network_'+self.name) #get the set of variables and then send to update target network
        self.trainables = tf.trainable_variables(scope='Graph_'+self.name)
        self.target_trainables=tf.trainable_variables(scope='Graph_'+self.name+'_target_network_'+self.name)

        #store the name and initial values for target network
        self.targetOps=network.updateTargetGraph(self.trainables,tau)
    # self.update_target_net()

        print("Agent network initialization complete, Agent name:",self.name)

    def update_target_net(self):
        # set target network's parameters to be the same as the primary network
        """
        Copies the model parameters of one estimator to another.
        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """

        network.updateTarget(self.targetOps, self.sess)

    def predict(self, s):
        # make the prediction
        action = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1, \
                                                               self.mainQN.batch_size: 1})

        return action

    def predict_softmax(self, s):
        Qdist = self.sess.run(self.mainQN.Qout, feed_dict={ \
            self.mainQN.scalarInput: [s], \
            self.mainQN.trainLength: 1, self.mainQN.batch_size: 1})
        return Qdist

    def get_rnn_state(self, s, state):
        state1 = self.sess.run(self.mainQN.rnn_state,
                               feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1, \
                                          self.mainQN.batch_size: 1})

        return state1

    def train(self, trainBatch, trace_length, batch_size):
        # use double DQN as the training step
        # Use main net to make a prediction

        Q1 = self.sess.run(self.mainQN.predict, feed_dict={ \
            self.mainQN.scalarInput: np.vstack(trainBatch[:, 3]), self.mainQN.trainLength: trace_length, \
            self.mainQN.batch_size: batch_size})
        # Use target network to evaluate outputred
        Q2 = self.sess.run(self.targetQN.Qout, feed_dict={ \
            self.targetQN.scalarInput: np.vstack(trainBatch[:, 3]), \
            self.targetQN.trainLength: trace_length, self.targetQN.batch_size: batch_size})

        # Metl the Q value to obtain the target Q value
        doubleQ = Q2[range(batch_size * trace_length), Q1]
        targetQ = trainBatch[:, 2] + (.99 * doubleQ)  # .99 is the discount for doubleQ value

        # update network parameters with the predefined training method
        self.sess.run(self.mainQN.updateModel, \
                      feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ, \
                                 self.mainQN.actions: trainBatch[:, 1], self.mainQN.trainLength: trace_length, \
                                 self.mainQN.batch_size: batch_size})


    def per_train(self, trainBatch, trace_length, batch_size,ISWeights):
        # use double DQN as the training step
        # Use main net to make a prediction

        Q1 = self.sess.run(self.mainQN.predict, feed_dict={ \
            self.mainQN.scalarInput: np.vstack(trainBatch[:, 3]), self.mainQN.trainLength: trace_length, \
            self.mainQN.batch_size: batch_size})
        # Use target network to evaluate outputred
        Q2 = self.sess.run(self.targetQN.Qout, feed_dict={ \
            self.targetQN.scalarInput: np.vstack(trainBatch[:, 3]), \
            self.targetQN.trainLength: trace_length, self.targetQN.batch_size: batch_size})

        # Metl the Q value to obtain the target Q value
        doubleQ = Q2[range(batch_size * trace_length), Q1]
        targetQ = trainBatch[:, 2] + (.99 * doubleQ)  # .99 is the discount for doubleQ value

        # update network parameters with the predefined training method
        abs_error, _ = self.sess.run([self.mainQN.abs_errors, self.mainQN.updateModel],
                                    feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                               self.mainQN.targetQ: targetQ,
                                               self.mainQN.actions: trainBatch[:, 1],
                                               self.mainQN.trainLength: trace_length,
                                               self.mainQN.batch_size: batch_size,
                                                self.mainQN.ISWeights:ISWeights})

        return abs_error

    # remember the episodebuffer
    def remember(self, episodeBuffer):
        self.buffer.add(episodeBuffer)
