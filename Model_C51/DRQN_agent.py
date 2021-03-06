#Xinwu Qian 2019-02-07

#Agent file for agents who follow DRQN to update their rewards

import os
import numpy as np
import tensorflow as tf
import network
import time
import tensorflow.contrib.slim as slim
import config
from scipy.stats import norm






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

        self.count_single_act=0

        #QR params
        self.N=20; #number of quantiles
        self.k=1; #huber loss
        self.gamma=config.TRAIN_CONFIG['y']
        self.conf=1

        #risk averse
        # risk seeking behavior
        tmask = np.linspace(0, 1, num=self.N + 1)
        self.eta = config.NET_CONFIG['eta']
        # self.quantile_mask = tmask ** self.eta / (tmask ** self.eta + (1 - tmask) ** self.eta) ** (
        #             1 / self.eta)
        if config.NET_CONFIG['Risk_Distort']:
            self.quantile_mask=norm.cdf(norm.ppf(tmask)-self.eta)
        else:
            self.quantile_mask=tmask

        self.quantile_mask = np.diff(self.quantile_mask) # rescale the distribution to favor risk neutral or risk-averse behavior


        #risk seeking
        # self.mask=np.concatenate([np.zeros(self.N-self.N//2),np.ones(self.N//2)])
        # self.quantile_mask=self.mask

        #place holders.
        #for current observations
        self.scalarInput = tf.placeholder(shape=[None, N_station * N_station * 5], dtype=tf.float32,name='main_input')
        self.target_scalarInput= tf.placeholder(shape=[None, N_station * N_station * 5], dtype=tf.float32,name='target_input')
        self.trainLength = tf.placeholder(dtype=tf.int32, name= 'trainlength')
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name= 'batchsize')
        self.iter_holder=tf.placeholder(dtype=tf.float32,shape=[None, 1],name='iterholder')
        self.eps_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='epsholder')

        self.targetQ=[]
        self.actions=[]
        self.rewards=[]
        self.main_rnn_value=[]
        self.station_score=[]
        self.predict_score = []
        self.rnn_holder=tf.placeholder(shape=[None, self.lstm_units], dtype=tf.float32,name='main_input')
        self.rnn_cstate_holder = tf.placeholder(shape=[1,None,self.lstm_units], dtype=tf.float32, name='main_input')
        self.rnn_hstate_holder = tf.placeholder(shape=[1,None,self.lstm_units], dtype=tf.float32, name='main_input')

        for i in range(N_station):
            targetQ=tf.placeholder(shape=[None,self.N], dtype=tf.float32)
            actions=tf.placeholder(shape=[None], dtype=tf.int32)
            rewards=tf.placeholder(shape=[None],dtype=tf.float32)
            predict_score= tf.placeholder(dtype=tf.float32, shape=[None, self.N_station])
            self.targetQ.append(targetQ)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.predict_score.append(predict_score)

        #ops.
        self.mainQout=[]
        self.targetQout=[]
        self.mainPredict=[]
        self.updateModel=[]
        self.targetZ=[]


    def build_main(self):
        myScope_main = 'DRQN_Main_'
        imageIn = tf.reshape(self.scalarInput, shape=[-1, self.N_station, self.N_station, 5], name=myScope_main + 'in1')
        input_conv = tf.pad(imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT",
                            name=myScope_main + 'in2')  # reflect padding!
        conv2 = tf.layers.conv2d( \
            inputs=input_conv, filters=16, \
            kernel_size=[5, 5], strides=[2, 2], activation='relu',reuse=None,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),padding='VALID', \
            name=myScope_main + '_net_conv2')
        bn = tf.layers.batch_normalization(conv2, training=True)
        conv3 = tf.layers.conv2d( \
            inputs=bn, filters=32, \
            kernel_size=[3, 3], strides=[1, 1], activation='relu',reuse=None,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),padding='VALID', \
            name=myScope_main + '_net_conv3')
        bn = tf.layers.batch_normalization(conv3, training=True)

        if self.use_gpu:
            print('Using CudnnLSTM')
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_units, name=myScope_main + '_lstm')

        else:
            print('Using LSTMfused')
            lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope_main + '_lstm')


        convFlat = tf.reshape(slim.flatten(bn), [self.batch_size, self.trainLength, self.h_size],
                              name=myScope_main + '_convlution_flattern')

        iter=tf.reshape(self.iter_holder,[self.batch_size,self.trainLength,1])
        eps=tf.reshape(self.eps_holder,[self.batch_size,self.trainLength,1])
        convFlat=tf.concat([convFlat,iter,eps],axis=-1)

        # action_code= tf.reshape(tf.contrib.layers.flatten(tf.one_hot(self.action_holder, self.N_station, dtype=tf.float32)),[self.batch_size,self.trainLength,self.N_station**2]) # out: [None, n_actionsxn_actions]
        # convFlat=tf.concat([convFlat,action_code],axis=-1)
        rnn, rnn_state = lstm(inputs=convFlat)
        rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope_main + '_reshapeRNN_out')

        my_initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.rnn_cstate_holder,self.rnn_hstate_holder)
        rnnin,rnn_out_state=lstm(inputs=convFlat,initial_state=my_initial_state)
        self.main_rnn_value.append(rnnin)
        self.rnn_out_state=rnn_out_state
        streamA, streamV = tf.split(rnn, 2, 1, name=myScope_main + '_split_streamAV')


        for i in range(self.N_station):
            # The output from the recurrent player is then split into separate Value and Advantage streams
            myScope = 'DRQN_main_' + str(i)

            Advantage = tf.layers.dense(streamA, (self.N_station) * self.N, activation='linear',name=myScope + 'AW',
                                        reuse=None)  # advantage
            Value = tf.layers.dense(streamV, 1, name=myScope + 'VW',activation='linear', reuse=None)  # advantage

            Qt = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),
                                     name=myScope + '_unshaped_Qout')
            Qout = tf.reshape(Qt, [-1, self.N_station, self.N])  # reshape it to N_station by self.atoms dimension
            self.mainQout.append(Qout)

            # for prediction
            streamA2, streamV2 = tf.split(self.rnn_holder, 2, 1, name=myScope + '_split_streamAV')
            Advantage2 = tf.layers.dense(streamA2, (self.N_station) * self.N, activation='linear',name=myScope + 'AW',
                                         reuse=True)  # advantage
            Value2 = tf.layers.dense(streamV2, 1, name=myScope + 'VW', activation='linear',reuse=True)  # advantage
            Qt2 = Value2 + tf.subtract(Advantage2, tf.reduce_mean(Advantage2, axis=1, keepdims=True),
                                       name=myScope + '_unshaped_Qout')
            Qout2 = tf.reshape(Qt2, [-1, self.N_station, self.N])  # reshape it to N_station by self.atoms dimension
            #
            q = tf.reduce_mean(tf.sort(Qout2, axis=-1) * self.quantile_mask, axis=-1)
            station_vec = tf.concat([tf.ones(i), tf.zeros(1), tf.ones(self.N_station - i - 1)], axis=0)
            station_score = tf.multiply(self.predict_score[i], station_vec)  # mark self as 0
            self.station_score.append(station_score)
            # predict based on the 95% confidence interval
            # predict = tf.argmax(tf.subtract(tf.add(mean,tf.scalar_mul(self.conf,std)),self.station_score[i]), 1, name=myScope + '_prediction')
            predict = tf.argmax(tf.subtract(q, self.station_score[i]), 1, name=myScope + '_prediction')
            self.mainPredict.append(predict)



    def build_target(self):
        myScope_main = 'DRQN_Target_'
        imageIn = tf.reshape(self.scalarInput, shape=[-1, self.N_station, self.N_station, 5], name=myScope_main + 'in1')
        input_conv = tf.pad(imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT",
                            name=myScope_main + 'in2')  # reflect padding!
        conv2 = tf.layers.conv2d( \
            inputs=input_conv, filters=16, \
            kernel_size=[5, 5], strides=[2, 2], activation='relu',reuse=None,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),padding='VALID', \
            name=myScope_main + '_net_conv2')
        bn = tf.layers.batch_normalization(conv2, training=True)
        conv3 = tf.layers.conv2d( \
            inputs=bn, filters=32, \
            kernel_size=[3, 3], strides=[1, 1], activation='relu',reuse=None,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),padding='VALID', \
            name=myScope_main + '_net_conv3')
        bn = tf.layers.batch_normalization(conv3, training=True)

        if self.use_gpu:
            print('Using CudnnLSTM')
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.lstm_units, name=myScope_main + '_lstm')

        else:
            print('Using LSTMfused')
            lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.lstm_units, name=myScope_main + '_lstm')


        convFlat = tf.reshape(slim.flatten(bn), [self.batch_size, self.trainLength, self.h_size],
                              name=myScope_main + '_convlution_flattern')

        iter=tf.reshape(self.iter_holder,[self.batch_size,self.trainLength,1])
        eps=tf.reshape(self.eps_holder,[self.batch_size,self.trainLength,1])
        convFlat=tf.concat([convFlat,iter,eps],axis=-1)
        # action_code= tf.reshape(tf.contrib.layers.flatten(tf.one_hot(self.action_holder, self.N_station, dtype=tf.float32)),[self.batch_size,self.trainLength,self.N_station**2]) # out: [None, n_actionsxn_actions]
        # convFlat=tf.concat([convFlat,action_code],axis=-1)
        rnn, rnn_state = lstm(inputs=convFlat)
        rnn = tf.reshape(rnn, shape=[-1, self.lstm_units], name=myScope_main + '_reshapeRNN_out')
        streamA, streamV = tf.split(rnn, 2, 1, name=myScope_main + '_split_streamAV')

        for i in range(self.N_station):
            myScope = 'DRQN_Target_' + str(i)
            Advantage = tf.layers.dense(streamA, (self.N_station) * self.N, activation='linear',name=myScope + 'AW',
                                        reuse=None)  # advantage
            Value = tf.layers.dense(streamV, 1, name=myScope + 'VW',activation='linear', reuse=None)  # advantage

            Qt = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True),
                                     name=myScope + '_unshaped_Qout')
            Qout = tf.reshape(Qt, [-1, self.N_station, self.N])  # reshape it to N_station by self.atoms dimension
            self.targetQout.append(Qout)


    def build_train(self):
        masklength=3
        mask1=tf.zeros([self.batch_size,masklength])
        mask2= tf.ones([self.batch_size, self.train_length-masklength])
        mask=tf.concat([mask1,mask2],axis=-1)
        self.mask = tf.reshape(mask, [-1])
        trainer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam_opt')
        for i in range(self.N_station):
            myScope = 'nothing'+str(i)
            q = tf.reduce_mean(tf.sort(self.mainQout[i], axis=-1) * self.quantile_mask, axis=-1)
            main_q = tf.subtract(q, self.station_score[i])
            main_act = tf.argmax(main_q, axis=-1)

            # Return the evaluation from target network
            target_mask = tf.one_hot(main_act, self.N_station, dtype=tf.float32)  # out: [None, n_actions]
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
            updateModel = trainer.minimize(loss, name=myScope + '_training')
            self.updateModel.append(updateModel)

    def drqn_build(self):
        self.build_main()
        self.build_target()
        self.build_train()
        # self.main_trainables = tf.trainable_variables(scope='DRQN_main_')
        self.trainables = tf.trainable_variables(scope='DRQN')
        # self.target_trainables = tf.trainable_variables(scope='DRQN_target')

        # store the name and initial values for target network
        self.targetOps = network.updateTargetGraph(self.trainables, self.tau)
        # self.update_target_net()

        print("Agent network initialization complete with:",str(self.N_station),' agents')



    def update_target_net(self):
        network.updateTarget(self.targetOps, self.sess)


    def predict(self, rnn,predict_score,e,station,threshold,rng,valid,invalid):
        # make the prediction
        valid[station]=True

       # print(self.conf)
        if rng<e: #epsilon greedy
            if e==1:
                action=np.random.randint(len(predict_score))
            else:
                idx=[i for i, x in enumerate(valid) if x]
                if station not in idx:
                    idx.append(station)
                action=np.random.choice(idx)

        else:
            #get the adjusted predict score
            #print(predict_score)
            # if sum(b)<=2:
            #     b=adj_predict.argsort()[-3:][::-1]
            # print('Feasible Solution:',sum(b), 'Station ID:',station)
            predict_score[invalid]=1e4
            predict_score[valid]=0;
            Q= self.sess.run(self.mainPredict[station], feed_dict={self.rnn_holder: rnn[0][0], self.predict_score[station]:[predict_score]})
            action=Q[-1]

        return action


    def batch_predict(self,t1,l):
        #return the action of each station
        v=tf.gather(self.mainPredict,t1)
        l=tf.concat([l,v],0)
        return t1+1,l

    def condition(self,t1,l):
        return t1<self.N_station

    def train_prepare(self, trainBatch,station):

        #params:
        # Q1 and Q2 in the shape of [batch*length, N_station, N]
        reward=np.array([r[station] for r in trainBatch[:,2]])
        action=np.array([a[station] for a in trainBatch[:,1]])
        #reward[action==self.N_station]=0;
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
        a_mask = tf.one_hot(action, self.N_station, dtype=tf.float32)  # out: [None, n_actions]
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

        target_mask = tf.one_hot(main_act, self.N_station, dtype=tf.float32)  # out: [None, n_actions]
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
            huber_loss = tf.where(be<self.k,0.5*tf.square(be),self.k*(be-0.5*self.k))
            quantile_loss = quant_weight * huber_loss  # out: [None, N, N]

        quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)  # Expected loss for each quantile
        loss = tf.reduce_sum(quantile_loss, axis=-1)  # Sum loss over all quantiles

        #hysteria loss
        #loss_hyst=tf.where(tf.reduce_mean(targetQ,axis=-1)<tf.reduce_mean(mainQ,axis=-1),0.3*loss,loss)
        # loss = tf.reduce_mean(loss)  # Average loss over the batch

        return loss

    def update_eta(self,target_eta,iteration):
        stepDrop = (1- target_eta) / iteration;
        self.eta-=stepDrop
        if self.eta<target_eta:
            self.eta=target_eta;
        mask=np.linspace(0, 1, num=self.N + 1)
        self.quantile_mask = mask ** self.eta / (mask ** self.eta + (1 - mask) ** self.eta) ** (1 / self.eta)
        self.quantile_mask = np.diff(self.quantile_mask) # rescale the distribution to favor risk neutral or risk-averse behavior

    def update_conf(self,var,iteration):
        stepDrop = (var) / (0.5*iteration);
        self.conf-=stepDrop
        if self.conf<-var:
            self.conf=-var



