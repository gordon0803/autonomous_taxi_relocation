#Xinwu Qian 2019-02-07

#Agent file for agents who follow DRQN to update their rewards

import os
import numpy as np
import tensorflow as tf
import network
import time
import tensorflow.contrib.slim as slim




#lets define a memory efficient drqn_agent()

class drqn_agent_efficient():
    def __init__(self,N_station,h_size,tau,sess,batch_size,train_length,is_gpu=0,ckpt_path=None):
        self.N_station=N_station;
        self.h_size=h_size;
        self.tau=tau;
        self.sess=sess;
        self.train_length=train_length;
        self.use_gpu=is_gpu;
        self.ckpt_path=ckpt_path;

        #place holders.
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 6], dtype=tf.float32,name='main_input')
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, N_station, N_station, 6])
        self.trainLength = tf.placeholder(dtype=tf.int32, name= 'trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name= 'batchsize')

        self.targetQ=[]
        self.actions=[]

        for i in range(N_station):
            targetQ=tf.placeholder(shape=[None], dtype=tf.float32)
            actions=tf.placeholder(shape=[None], dtype=tf.int32)
            self.targetQ.append(targetQ)
            self.actions.append(actions)

        #nets
        # self.conv1=[]
        # self.conv2=[]
        # self.conv3=[]
        # self.conv4=[]
        # self.rnn=[]

        #ops.
        self.mainQout=[]
        self.targetQout=[]
        self.mainPredict=[]
        self.updateModel=[]


    def build_main(self):
        ##construct the main network
        maskA = tf.zeros([self.batch_size, self.train_length // 2])  # Mask first 20 records are shown to have the best results
        maskB = tf.ones([self.batch_size, self.train_length // 2])
        mask = tf.concat([maskA, maskB], 1)
        self.mask = tf.reshape(mask, [-1])
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam_opt')

        for i in range(self.N_station):
            myScope = 'DRQN_main_'+str(i)
            conv1 = tf.nn.relu(tf.layers.conv2d( \
                inputs=self.imageIn, filters=64, \
                kernel_size=[3, 3], strides=[2, 2], padding='VALID', \
                name=myScope + '_net_conv1'))
            conv2 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv1, filters=64, \
                 kernel_size=[2, 2], strides=[2, 2], padding='VALID', \
                 name=myScope + '_net_conv2'))
            convFlat = tf.reshape(slim.flatten(conv2), [self.batch_size, self.trainLength, self.h_size],
                                       name=myScope + '_convlution_flattern')
            if self.use_gpu:
                print('Using CudnnLSTM')
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.h_size, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat)
            else:
                print('Using LSTMfused')
                lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.h_size, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat, dtype=tf.float32)

            rnn = tf.reshape(rnn, shape=[-1, self.h_size], name=myScope + '_reshapeRNN_out')
            # The output from the recurrent player is then split into separate Value and Advantage streams
            streamA, streamV = tf.split(rnn, 2, 1, name=myScope + '_split_streamAV')
            AW = tf.Variable(tf.random_normal([self.h_size // 2, self.N_station + 1]), name=myScope + 'AW')  # action +1, with the last action being station without any vehicles

            VW = tf.Variable(tf.random_normal([self.h_size //2, 1]), name=myScope + 'VW')

            Advantage = tf.matmul(streamA, AW, name=myScope + '_matmulAdvantage')
            Value = tf.matmul(streamV, VW, name=myScope + '_matmulValue')

            Qout = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),name=myScope + '_Qout')
            self.mainQout.append(Qout)
            predict = tf.argmax(Qout, 1, name=myScope + '_prediction')
            self.mainPredict.append(predict)
            actions_onehot = tf.one_hot(self.actions[i], self.N_station + 1, dtype=tf.float32,  name = myScope + '_onehot')  # action +1, with the last action being station without any vehicles

            salience = tf.gradients(Advantage, self.imageIn)
            # Then combine them together to get our final Q-values.
            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            Q = tf.reduce_sum(tf.multiply(Qout, actions_onehot), axis=1, name=myScope + 'Qvalue')
            base_error=tf.abs(self.targetQ[i]-Q)
            clipv=10;
            huber_error=tf.where(base_error<clipv,0.5*tf.square(base_error),clipv*(base_error-0.5*clipv),name=myScope+'_HuberError')
            td_error = tf.square(self.targetQ[i] - Q, name=myScope + '_TDERROR')
            hyst_error=tf.where(self.targetQ[i]-Q<0,0.4*huber_error,huber_error,name=myScope+'_hysterError')
            # In order to only propogate accurate gradients through the network, we will mask the first
            # half of the losses for each trace as per Lample & Chatlot 2016
            loss = tf.reduce_mean(hyst_error * self.mask, name=myScope + '_defineloss')
            updateModel = self.trainer.minimize(loss, name=myScope + '_training')
            self.updateModel.append(updateModel)

    def build_target(self):
        ##construct the main network
        for i in range(self.N_station):
            myScope = 'DRQN_target_' + str(i)
            conv1 = tf.nn.relu(tf.layers.conv2d( \
                inputs=self.imageIn, filters=64, \
                kernel_size=[3, 3], strides=[2, 2], padding='VALID', \
                name=myScope + '_net_conv1'))
            conv2 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv1, filters=64, \
                 kernel_size=[2, 2], strides=[2, 2], padding='VALID', \
                 name=myScope + '_net_conv2'))
            convFlat = tf.reshape(slim.flatten(conv2), [self.batch_size, self.trainLength, self.h_size],
                                  name=myScope + '_convlution_flattern')
            if self.use_gpu:
                print('Using CudnnLSTM')
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.h_size, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat)
            else:
                print('Using LSTMfused')
                lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.h_size, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat, dtype=tf.float32)

            rnn = tf.reshape(rnn, shape=[-1, self.h_size], name=myScope + '_reshapeRNN_out')
            # The output from the recurrent player is then split into separate Value and Advantage streams
            streamA, streamV = tf.split(rnn, 2, 1, name=myScope + '_split_streamAV')
            AW = tf.Variable(tf.random_normal([self.h_size // 2, self.N_station + 1]),
                             name=myScope + 'AW')  # action +1, with the last action being station without any vehicles

            VW = tf.Variable(tf.random_normal([self.h_size // 2, 1]), name=myScope + 'VW')

            Advantage = tf.matmul(streamA, AW, name=myScope + '_matmulAdvantage')
            Value = tf.matmul(streamV, VW, name=myScope + '_matmulValue')

            Qout = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),
                                       name=myScope + '_Qout')
            self.targetQout.append(Qout)

    def drqn_build(self):
        self.build_main()
        self.build_target()
        self.main_trainables = tf.trainable_variables(scope='DRQN_main_')
        self.trainables = tf.trainable_variables(scope='DRQN')
        self.target_trainables = tf.trainable_variables(scope='DRQN_target')

        # store the name and initial values for target network
        self.targetOps = network.updateTargetGraph(self.trainables, self.tau)
        # self.update_target_net()

        print("Agent network initialization complete with:",str(self.N_station),' agents')


    def update_target_net(self):
        network.updateTarget(self.targetOps, self.sess)


    def predict(self, s,predict_score,e,station):
        # make the prediction
        threshold=0.4
        legible = predict_score >= threshold

        if np.random.rand(1)<e: #epsilon greedy
            idx=[i for i, x in enumerate(legible) if x]
            if station not in idx:
                idx.append(station)
            action=np.random.choice(idx)

        else:
            Qvalue = self.sess.run(self.mainQout[station], feed_dict={self.scalarInput: [s], self.trainLength: 1, self.batch_size: 1})
            Qvalue=np.array(Qvalue[0])
            legible=np.append(predict_score>threshold,True)
            legible[station]=True
            action=np.argmax(Qvalue*legible)

        return action

    def predict_regular(self, s,station):
        action = self.sess.run(self.mainPredict[station], feed_dict={self.scalarInput: [s], self.trainLength: 1,
                                                            self.batch_size: 1})
        return action

    def train(self, trainBatch, trace_length, batch_size,linear_model,e,station,N_station,predict_score):
        # use double DQN as the training step
        # Use main net to make a prediction

        Q1 = self.sess.run(self.mainQout[station], feed_dict={ \
            self.scalarInput: np.vstack(trainBatch[:, 3]), self.trainLength: trace_length, \
            self.batch_size: batch_size})

        #prediction based on the optimal feasible values.
        legible = predict_score >= 0.4
        legible_true=np.ones((batch_size*trace_length,N_station+1))
        legible_true[:,:-1]=legible #assign value
        legible_true[:,station] = True #change column value to legible solutions
        Q1 = np.argmax(Q1 * legible_true, axis=1)

        # Use target network to evaluate outputred
        Q2 = self.sess.run(self.targetQout[station], feed_dict={ \
            self.scalarInput: np.vstack(trainBatch[:, 3]), \
            self.trainLength: trace_length, self.batch_size: batch_size})

        # Metl the Q value to obtain the target Q value

        doubleQ = Q2[range(batch_size * trace_length), Q1]

        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        targetQ = reward + (.8 * doubleQ)  # .99 is the discount for doubleQ value


        # update network parameters with the predefined training method
        self.sess.run(self.updateModel[station], \
                      feed_dict={self.scalarInput: np.vstack(trainBatch[:, 0]), self.targetQ[station]: targetQ, \
                                 self.actions[station]: action, self.trainLength: trace_length, \
                                 self.batch_size: batch_size})

    def train_prepare(self, trainBatch, trace_length, batch_size,linear_model,e,station,N_station,predict_score,Q1,Q2,use_linear):
        if use_linear:
            legible = predict_score >= 0.4
            legible_true=np.ones((batch_size*trace_length,N_station+1))
            legible_true[:,:-1]=legible #assign value
            legible_true[:,station] = True #change column value to legible solutions
            Q1 = np.argmax(Q1 * legible_true, axis=1)
        else:
            Q1=np.argmax(Q1,axis=1) #take the maximum operation

        doubleQ = Q2[range(batch_size * trace_length), Q1]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        targetQ = reward + (.8 * doubleQ)  # .99 is the discount for doubleQ value
        return targetQ,action


    @staticmethod
    def get_threshold(e,threshold):
        if e>1-threshold:
            threshold=1-e
        return threshold



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
        network.updateTarget(self.targetOps, self.sess)

    def predict(self, s,predict_score,e,station):
        # make the prediction
        if e>0.8:
            threshold=1-e
        else:
            threshold=0.2;
        legible = predict_score >= threshold

        if np.random.rand(1)<e: #epsilon greedy
            idx=[i for i, x in enumerate(legible) if x]
            if station not in idx:
                idx.append(station)
            action=np.random.choice(idx)

        else:
            Qvalue = self.sess.run(self.mainQN.Qout, feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1, self.mainQN.batch_size: 1})
            Qvalue=np.array(Qvalue[0])
            legible=np.append(predict_score>threshold,True)
            legible[station]=True
            action=np.argmax(Qvalue*legible)

        return action

    def predict_regular(self, s):
        action = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1,
                                                            self.mainQN.batch_size: 1})
        return action


    def get_rnn_state(self, s, state):
        state1 = self.sess.run(self.mainQN.rnn_state,
                               feed_dict={self.mainQN.scalarInput: [s], self.mainQN.trainLength: 1, \
                                          self.mainQN.batch_size: 1})

        return state1

    def train(self, trainBatch, trace_length, batch_size,linear_model,e,station,N_station,predict_score):
        # use double DQN as the training step
        # Use main net to make a prediction

        Q1 = self.sess.run(self.mainQN.Qout, feed_dict={ \
            self.mainQN.scalarInput: np.vstack(trainBatch[:, 3]), self.mainQN.trainLength: trace_length, \
            self.mainQN.batch_size: batch_size})

        #prediction based on the optimal feasible values.
        legible = predict_score >= 0.4
        legible_true=np.ones((batch_size*trace_length,N_station+1))
        legible_true[:,:-1]=legible #assign value
        legible_true[:,station] = True #change column value to legible solutions
        Q1 = np.argmax(Q1 * legible_true, axis=1)

        # Use target network to evaluate outputred
        Q2 = self.sess.run(self.targetQN.Qout, feed_dict={ \
            self.targetQN.scalarInput: np.vstack(trainBatch[:, 3]), \
            self.targetQN.trainLength: trace_length, self.targetQN.batch_size: batch_size})

        # Metl the Q value to obtain the target Q value

        doubleQ = Q2[range(batch_size * trace_length), Q1]

        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        targetQ = reward + (.8 * doubleQ)  # .99 is the discount for doubleQ value


        # update network parameters with the predefined training method
        self.sess.run(self.mainQN.updateModel, \
                      feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ, \
                                 self.mainQN.actions: action, self.mainQN.trainLength: trace_length, \
                                 self.mainQN.batch_size: batch_size})


    def train_prepare(self, trainBatch, trace_length, batch_size,linear_model,e,station,N_station,predict_score,Q1,Q2,use_linear):
        if use_linear:
            legible = predict_score >= 0.4
            legible_true=np.ones((batch_size*trace_length,N_station+1))
            legible_true[:,:-1]=legible #assign value
            legible_true[:,station] = True #change column value to legible solutions
            Q1 = np.argmax(Q1 * legible_true, axis=1)
        else:
            Q1=np.argmax(Q1,axis=1) #take the maximum operation

        doubleQ = Q2[range(batch_size * trace_length), Q1]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        targetQ = reward + (.8 * doubleQ)  # .99 is the discount for doubleQ value
        return targetQ,action




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
        targetQ = trainBatch[:, 2] + (.8 * doubleQ)  # .99 is the discount for doubleQ value

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

    @staticmethod
    def get_threshold(e,threshold):
        if e>1-threshold:
            threshold=1-e
        return threshold

