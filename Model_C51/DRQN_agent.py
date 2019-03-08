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
    def __init__(self,N_station,h_size,lstm_units,tau,sess,batch_size,train_length,is_gpu=0,ckpt_path=None):
        self.N_station=N_station;
        self.h_size=h_size;
        self.lstm_units=lstm_units;
        self.tau=tau;
        self.sess=sess;
        self.train_length=train_length;
        self.use_gpu=is_gpu;
        self.ckpt_path=ckpt_path;



        #QR params
        self.N=51; #number of quantiles
        self.k=1; #huber loss
        self.gamma=0.99**10 #discount factor

        #place holders.
        #for current observations
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 6], dtype=tf.float32,name='main_input')

        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, N_station, N_station, 6])
        self.input_conv = tf.pad(self.imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")  # reflect padding!


        self.trainLength = tf.placeholder(dtype=tf.int32, name= 'trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name= 'batchsize')
        self.targetQ=[]
        self.actions=[]
        self.rewards=[]
        self.station_score=[]
        self.predict_score = []

        for i in range(N_station):
            targetQ=tf.placeholder(shape=[None,self.N], dtype=tf.float32)
            actions=tf.placeholder(shape=[None], dtype=tf.int32)
            rewards=tf.placeholder(shape=[None],dtype=tf.float32)
            predict_score= tf.placeholder(dtype=tf.float32, shape=[None, self.N_station + 1])
            self.targetQ.append(targetQ)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.predict_score.append(predict_score)


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
        self.targetZ=[]


    def build_main(self):
        for i in range(self.N_station):
            myScope = 'DRQN_main_'+str(i)
            conv1 = tf.nn.relu(tf.layers.conv2d( \
                inputs=self.input_conv, filters=16, \
                kernel_size=[4, 4], strides=[1, 1], padding='VALID', \
                name=myScope + '_net_conv1'))
            conv2 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv1, filters=32, \
                 kernel_size=[4, 4], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv2'))
            conv3 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv2, filters=32, \
                 kernel_size=[3, 3], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv3'))
            conv4 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv3, filters=32, \
                 kernel_size=[3, 3], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv4'))
            convFlat = tf.reshape(slim.flatten(conv4), [self.batch_size, self.trainLength, self.h_size],
                                       name=myScope + '_convlution_flattern')
            if self.use_gpu:
                print('Using CudnnLSTM')
                lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.lstm_units, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat)
            else:
                print('Using LSTMfused')
                lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat, dtype=tf.float32)

            rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope + '_reshapeRNN_out')
            # The output from the recurrent player is then split into separate Value and Advantage streams
            streamA, streamV = tf.split(rnn, 2, 1, name=myScope + '_split_streamAV')
            AW = tf.Variable(tf.random_normal([self.lstm_units // 2, (self.N_station + 1)*self.N]), name=myScope + 'AW')  # action +1, with the last action being station without any vehicles

            VW = tf.Variable(tf.random_normal([self.lstm_units //2, 1]), name=myScope + 'VW')

            Advantage = tf.matmul(streamA, AW, name=myScope + '_matmulAdvantage')
            Value = tf.matmul(streamV, VW, name=myScope + '_matmulValue')

            Qt = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),name=myScope + '_unshaped_Qout')
            Qout=tf.reshape(Qt, [-1, self.N_station+1, self.N])  #reshape it to N_station + 1 by self.atoms dimension
            self.mainQout.append(Qout)

            q=tf.reduce_mean(Qout,axis=-1)
            station_vec=tf.concat([tf.ones(i),tf.zeros(1),tf.ones(self.N_station-i)],axis=0)
            station_score=tf.multiply(self.predict_score[i],station_vec)  #mark self as 0
            self.station_score.append(station_score)
            predict = tf.argmax(tf.subtract(q,self.station_score[i]), 1, name=myScope + '_prediction')
            self.mainPredict.append(predict)

    def build_target(self):
        ##construct the main network

        for i in range(self.N_station):
            myScope = 'DRQN_target_' + str(i)
            conv1 = tf.nn.relu(tf.layers.conv2d( \
                inputs=self.input_conv, filters=16, \
                kernel_size=[4, 4], strides=[1, 1], padding='VALID', \
                name=myScope + '_net_conv1'))
            conv2 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv1, filters=32, \
                 kernel_size=[4, 4], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv2'))
            conv3 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv2, filters=32, \
                 kernel_size=[3, 3], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv3'))
            conv4 = tf.nn.relu(tf.layers.conv2d( \
                 inputs=conv3, filters=32, \
                 kernel_size=[3, 3], strides=[1, 1], padding='VALID', \
                 name=myScope + '_net_conv4'))
            convFlat = tf.reshape(slim.flatten(conv4), [self.batch_size, self.trainLength, self.h_size],
                                       name=myScope + '_convlution_flattern')
            if self.use_gpu:
                print('Using CudnnLSTM')
                lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.lstm_units, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat)
            else:
                print('Using LSTMfused')
                lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope + '_lstm')
                rnn, rnn_state = lstm(inputs=convFlat, dtype=tf.float32)

            rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope + '_reshapeRNN_out')
            # The output from the recurrent player is then split into separate Value and Advantage streams
            streamA, streamV = tf.split(rnn, 2, 1, name=myScope + '_split_streamAV')
            AW = tf.Variable(tf.random_normal([self.lstm_units // 2, (self.N_station + 1) * self.N]),
                             name=myScope + 'AW')  # action +1, with the last action being station without any vehicles

            VW = tf.Variable(tf.random_normal([self.lstm_units // 2, 1]), name=myScope + 'VW')

            Advantage = tf.matmul(streamA, AW, name=myScope + '_matmulAdvantage')
            Value = tf.matmul(streamV, VW, name=myScope + '_matmulValue')

            Qt = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),name=myScope + '_unshaped_Qout')
            Qout=tf.reshape(Qt, [-1, self.N_station+1, self.N])  #reshape it to N_station + 1 by self.atoms dimension
            self.targetQout.append(Qout)


    def build_train(self):
        maskA = tf.zeros([self.batch_size, self.train_length//2])  # Mask first 20 records are shown to have the best results
        maskB = tf.ones([self.batch_size, self.train_length//2])
        mask = tf.concat([maskA, maskB], 1)
        self.mask = tf.reshape(mask, [-1])
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam_opt')
        for i in range(self.N_station):
            myScope = 'DRQN_main_'+str(i)
            # Then combine them together to get our final Q-values.
            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            main_q = tf.reduce_mean(self.mainQout[i], axis=-1)

            main_q = tf.subtract(main_q, self.station_score[i])
            main_act = tf.argmax(main_q, axis=-1)
            # Return the evaluation from target network
            target_mask = tf.one_hot(main_act, self.N_station + 1, dtype=tf.float32)  # out: [None, n_actions]
            target_mask = tf.expand_dims(target_mask, axis=-1)  # out: [None, n_actions, 1]
            selected_target= tf.reduce_sum(self.targetQout[i] * target_mask, axis=1)  # out: [None, N]

            rew_t = tf.expand_dims(self.rewards[i], axis=-1)
            target_z = rew_t + self.gamma * selected_target
            self.targetZ.append(target_z)

            mainz=self._compute_estimate(self.mainQout[i],self.actions[i])
            loss = self._compute_loss(mainz,self.targetQ[i])
            loss = tf.reduce_mean(loss * self.mask, name=myScope + '_maskloss')
            # In order to only propogate accurate gradients through the network, we will mask the first
            # half of the losses for each trace as per Lample & Chatlot 2016
            updateModel = self.trainer.minimize(loss, name=myScope + '_training')
            self.updateModel.append(updateModel)


    def drqn_build(self):
        self.build_main()
        self.build_target()
        self.build_train()
        self.main_trainables = tf.trainable_variables(scope='DRQN_main_')
        self.trainables = tf.trainable_variables(scope='DRQN')
        self.target_trainables = tf.trainable_variables(scope='DRQN_target')

        # store the name and initial values for target network
        self.targetOps = network.updateTargetGraph(self.trainables, self.tau)
        # self.update_target_net()

        print("Agent network initialization complete with:",str(self.N_station),' agents')


    def update_target_net(self):
        network.updateTarget(self.targetOps, self.sess)


    def predict(self, s,predict_score,e,station,dist,exp_dist,threshold):
        # make the prediction
        predict_score=(predict_score*exp_dist)/dist
        legible = predict_score >= threshold
        legible[station]=True

        if np.random.rand(1)<e: #epsilon greedy
            if e==1:
                action=np.random.randint(len(predict_score))
            else:
                idx=[i for i, x in enumerate(legible) if x]
                if station not in idx:
                    idx.append(station)
                action=np.random.choice(idx)

        else:
            #get the adjusted predict score
            #print(predict_score)
            adj_predict=predict_score
            adj_predict=np.append(adj_predict,0)
            a=adj_predict<threshold
            b=adj_predict>=threshold
            adj_predict[a]=100
            adj_predict[b]=0;
            adj_predict[station]=0;
            Q= self.sess.run(self.mainPredict[station], feed_dict={self.scalarInput: [s], self.trainLength: 1, self.batch_size: 1,self.predict_score[station]:[adj_predict]})
            action=Q[0]

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
        threshold = self.get_threshold((1 - e), 0.4)
        threshold=0.4
        legible = predict_score >= threshold
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

    def train_prepare(self, trainBatch,station):

        #params:
        # Q1 and Q2 in the shape of [batch*length, N_station, N]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        return reward,action


    def _compute_estimate(self, agent_net,action):
        """Select the return distribution Z of the selected action
        Args:
          agent_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the agent
            action: 'tf.Tensor', shape '[None]
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """
        a_mask = tf.one_hot(action, self.N_station+1, dtype=tf.float32)  # out: [None, n_actions]
        a_mask = tf.expand_dims(a_mask, axis=-1)  # out: [None, n_actions, 1]
        z = tf.reduce_sum(agent_net * a_mask, axis=1)  # out: [None, N]
        return z


    def _select_target(self, main_out,target_out,predict_score):
        """Select the QRDQN target distributions - use the greedy action from E[Z]
        Args:
        main_out: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the main network

          target_out: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
            for the target network
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """

        #Choose the action from main network
        main_z=main_out
        main_q=tf.reduce_mean(main_z,axis=-1)
        main_q=tf.subtract(main_q,predict_score)
        main_act=tf.argmax(main_q,axis=-1)

        #Return the evaluation from target network

        target_mask = tf.one_hot(main_act, self.N_station+1, dtype=tf.float32)  # out: [None, n_actions]
        target_mask = tf.expand_dims(target_mask, axis=-1)  # out: [None, n_actions, 1]
        target_z = tf.reduce_sum(target_out * target_mask, axis=1)  # out: [None, N]
        return target_z


    def _compute_backup(self, target,reward):
        """Compute the QRDQN backup distributions
        Args:
          target: `tf.Tensor`, shape `[None, N]. The output from `self._select_target()`
        Returns:
          `tf.Tensor` of shape `[None, N]`
        """
        # Compute the projected quantiles; output shape [None, N]
        rew_t = tf.expand_dims(reward, axis=-1)
        target_z = rew_t + self.gamma * target
        return target_z

    def _compute_loss(self, mainQ, targetQ):
        """Compute the QRDQN loss.
        Args:
          mainQ: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_estimate()`
          targetQ: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_backup()`
        Returns:
          `tf.Tensor` of scalar shape `()`
        """

        # Compute the tensor of mid-quantiles
        mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
        mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
        mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

        # Operate over last dimensions to average over samples (target locations)
        td_z = tf.expand_dims(targetQ, axis=-2) - tf.expand_dims(mainQ, axis=-1)
        # td_z[0] =
        # [ [tz1-z1, tz2-z1, ..., tzN-z1],
        #   [tz1-z2, tz2-z2, ..., tzN-z2],
        #   ...
        #   [tz1-zN, tzN-zN, ..., tzN-zN]  ]
        indicator_fn = tf.to_float(td_z < 0.0)  # out: [None, N, N]

        # Compute the quantile penalty weights
        quant_weight = mid_quantiles - indicator_fn  # out: [None, N, N]
        # Make sure no gradient flows through the indicator function. The penalty is only a scaling factor
        quant_weight = tf.stop_gradient(quant_weight)

        # Pure Quantile Regression Loss
        if self.k == 0:
            quantile_loss = quant_weight * td_z  # out: [None, N, N]
        # Quantile Huber Loss
        else:
            quant_weight = tf.abs(quant_weight)
            be=tf.abs(td_z)
            huber_loss = tf.where(be<1,0.5*tf.square(be),be-0.5)
            quantile_loss = quant_weight * huber_loss  # out: [None, N, N]

        quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)  # Expected loss for each quntile
        loss = tf.reduce_sum(quantile_loss, axis=-1)  # Sum loss over all quantiles
        # loss = tf.reduce_mean(loss)  # Average loss over the batch

        return loss

    @staticmethod
    def get_threshold(e,threshold):
        if e<threshold:
            threshold=e
        return threshold

