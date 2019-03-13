import numpy as np
import time
import numba as nb
import scipy as sp
# @nb.jit
# def _return_upper_bound_batch(features,theta,alpha,AaI,n_action):
#     prob=[0 for i in range(len(features))]
#     count=0;
#     for feature in features:
#         tprob=[feature.dot(theta[i])+alpha*np.sqrt((feature.dot(AaI[i])).dot(feature)) for i in range(n_action)]
#         prob[count]=np.array(tprob)
#         count+=1
#     return np.array(prob)

# @nb.jit
# def _update_model(features,theta,AaI,n_action,actions,rewards,Da,ba,d,rounds):
#     for i in range(len(features)):
#         f = features[i]
#         for j in range(n_action):
#             action = actions[i][j]
#             if action < n_action:
#                 Da[action] += np.outer(f, f)
#                 ba[action] += rewards[i][action] * f
#
#     # inverse doesn't have to be calculated for each feature
#     for action in range(n_action):
#         AaI[action] = np.linalg.inv(np.identity(d) + Da[action])  # inverse
#         theta[action] = AaI[action].dot(ba[action])
#
#     rounds += len(features)  # number of rounds the bandit has been played
#     alpha = np.sqrt(0.5 * np.log(2 *rounds * n_action * 10))
#     return Da,ba,AaI,theta,rounds,alpha


class linucb_agnet():
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
        self.Da=[]
        self.theta=[]

        #initialize parameters
        for i in range(n_action):
            self.Aa.append(np.identity(self.d))
            self.ba.append(np.zeros(self.d))
            self.AaI.append(np.identity(self.d))
            self.Da.append(np.zeros((self.d,self.d)))
            self.theta.append(np.zeros(self.d))

    def update(self,features,actions,rewards):
        #update all observed arms

        # Da, ba, AaI, theta, rounds, alpha=_update_model(features,self.theta,self.AaI,self.n_action,actions,rewards,self.Da,self.ba,self.d,self.round)
        # self.Da=Da
        # self.ba=ba
        # self.AaI=AaI
        # self.theta=theta
        # self.round=rounds
        # self.alpha=alpha

        #reset parameters
        for i in range(len(features)):
            f=features[i]
            for j in range(self.n_action):
                action=actions[i][j]
                if action<self.n_action:
                    self.Da[action]+=np.outer(f,f)
                    self.ba[action]+= rewards[i][action]*f

        #inverse doesn't have to be calculated for each feature
        for action in range(self.n_action):
            self.AaI[action]=sp.linalg.inv(np.identity(self.d)+self.Da[action]) #inverse
            self.theta[action]=np.dot(self.AaI[action],self.ba[action])

        self.round+=len(features) #number of rounds the bandit has been played
        self.alpha=np.sqrt(0.5*np.log(2*self.round*self.n_action*10))

        #done update

    def return_upper_bound(self,feature):
        # prob=[]
        #
        # for i in range(self.n_action):
        #     score=np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s))
        #     prob.append(score)
        s=feature
        # prob=[np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s)) for i in range(self.n_action)]

        prob=np.fromiter((np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s)) for i in range(self.n_action)), float)
        # prob=_return_upper_bound(s,self.theta,self.alpha,self.AaI,self.n_action)

        return prob

    def return_upper_bound_batch(self,feature):
        t1=time.time()
        prob=[self.return_upper_bound(feature[i]) for i in range(len(feature))]

        return np.array(prob)





