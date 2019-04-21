import numpy as np
import time
import scipy
import numba as nb


# @nb.jit
# def _return_upper_bound_batch(features,theta,alpha,AaI,n_action):
#     prob=[0 for i in range(len(features))]
#     count=0;
#     for feature in features:
#         tprob=[feature.dot(theta[i])+alpha*np.sqrt((feature.dot(AaI[i])).dot(feature)) for i in range(n_action)]
#         prob[count]=np.array(tprob)
#         count+=1
#     return np.array(prob)


class linucb_agent():
    def __init__(self,n_action,d):
        #number of actions: n_action
        #feature length: d
        self.alpha=2
        self.round=0;
        self.d=d
        self.n_action=n_action
        self.Aa=[] #collection of A for each arm
        self.ba=[] #collection of vectors to compute disjoint part d*1
        self.AaI=[] #inverse of A
        self.theta=[]

        #initialize parameters
        for i in range(n_action):
            self.Aa.append(0.1*np.identity(self.d))
            self.ba.append(np.zeros(self.d))
            self.AaI.append(np.identity(self.d))
            self.theta.append(np.zeros(self.d))

    def update(self,features,actions,rewards):
        #update all observed arms
        #reset parameters
        #gamma=1; #decay parameter
       # for i in range(self.n_action):
        #    self.Da[i]=gamma*self.Da[i]
         #   self.ba[i]=gamma*self.ba[i]
        self.round+=len(rewards) #number of rounds the bandit has been played


        t1=time.time()
        features=np.vstack(features)
        actions=np.vstack(actions)
        rewards=np.vstack(rewards)
        features_exp=np.tile(features,[self.n_action,1]) #expand the features map
        actions_exp=actions.T.flatten()
        seq_id=np.tile([i for i in range(len(features))],self.n_action)
        rewards_exp=rewards[seq_id,actions_exp]

        for j in range(self.n_action):
            feature = features_exp[actions_exp == j]
            reward = rewards_exp[actions_exp == j]
            outer_feature = feature.T.dot(feature)
            outer_ba = feature.T .dot(reward)
            self.Aa[j] += outer_feature
            self.ba[j] += outer_ba
        # print('process 1:', time.time()-t1)

        # t1 = time.time()
        # for i in range(len(features)):
        #     f=features[i]
        #     for j in range(self.n_action): #each station actions
        #         action=actions[i][j]
        #         if action<self.n_action:
        #             self.Aa[action]+=f[:,None]*f[None,:]
        #             self.ba[action]+= rewards[i][action]*f
        # print('process 2:', time.time() - t1)
        #inverse doesn't have to be calculated for each feature
        for action in range(self.n_action):
            self.AaI[action]=scipy.linalg.pinv2(np.identity(self.d)+self.Aa[action]) #inverse
            self.theta[action]=np.dot(self.AaI[action],self.ba[action])


        # self.alpha=0.01
        self.alpha=np.sqrt(0.5*np.log(2*self.round*self.n_action*10))
        # self.alpha=0.01
        #print(self.alpha,self.round)

        #done update

    def return_upper_bound(self,feature):
        # prob=[]
       # s=np.array(feature)
        #
        # for i in range(self.n_action):
        #     score=np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s))
        #     prob.append(score)
        s=feature
        # prob=[np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s)) for i in range(self.n_action)]

        prob=np.fromiter((np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s)) for i in range(self.n_action)), float)


        # prob=_return_upper_bound(s,self.theta,self.alpha,self.AaI,self.n_action)

        return np.array(prob)

    def return_upper_bound_batch(self,feature):
        feature=np.vstack(feature)
        # prob=np.array([np.einsum('ij,j->i',feature,self.theta[i])+self.alpha*np.sqrt(np.einsum('ij,jj,ij->i',feature,self.AaI[i],feature)) for i in range(self.n_action)]).T
        prob = np.array([np.einsum('ij,j->i', feature, self.theta[i]) + self.alpha * np.sqrt(np.einsum('ij,jj,ij->i', feature, self.AaI[i], feature)) for i in range(self.n_action)]).T


        # prob=[self.return_upper_bound(feature[i]) for i in range(len(feature))]



        return np.array(prob)






