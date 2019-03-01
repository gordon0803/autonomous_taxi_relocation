import numpy as np

class linucb_agnet():
    def __init__(self,n_action,d):
        #number of actions: n_action
        #feature length: d
        self.alpha=1
        self.d=d
        self.n_action=n_action
        self.Aa={} #collection of A for each arm
        self.ba={} #collection of vectors to compute disjoint part d*1
        self.AaI={} #inverse of A
        self.Da={}
        self.theta={}

        #initialize parameters
        for i in range(n_action):
            self.Aa[i]=np.identity(self.d)
            self.ba[i]=np.zeros(self.d)
            self.AaI[i]=np.identity(self.d)
            self.Da[i]=np.zeros((self.d,self.d))
            self.theta[i]=np.zeros(self.d)

    def update(self,features,actions,rewards):
        #update all observed arms
        #reset parameters
        gamma=0.6; #decay parameter
        for i in range(self.n_action):
            self.Da[i]=gamma*self.Da[i]
            self.ba[i]=gamma*self.ba[i]
        for i in range(len(features)):
            f=np.array(features[i])
            for j in range(self.n_action):
                action=actions[i][j]
                if action<self.n_action:
                    self.Da[action]+=np.outer(f,f)
                    self.ba[action]+= rewards[i][action]*f

        #inverse doesn't have to be calculated for each feature
        for action in range(self.n_action):
            self.AaI[action]=np.linalg.inv(np.identity(self.d)+self.Da[action]) #inverse
            self.theta[action]=np.dot(self.AaI[action],self.ba[action])

        #done update

    def return_upper_bound(self,feature):
        prob=[]
        s=np.array(feature)
        for i in range(self.n_action):
            score=np.dot(s,self.theta[i])+self.alpha*np.sqrt(np.dot(np.dot(s,self.AaI[i]),s))
            prob.append(score)
        return np.array(prob)

    def return_upper_bound_batch(self,feature):
        prob=[]
        for i in range(len(feature)):
            prob.append(self.return_upper_bound(feature[i]))

        return np.array(prob)






