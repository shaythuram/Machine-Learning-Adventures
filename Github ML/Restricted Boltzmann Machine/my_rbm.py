# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


#importing datasets
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#treating data
training_set = pd.read_csv('ml-100k/u2.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u2.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#get the total number of movies n users
combined = np.append(training_set, test_set, axis=0)
nb_users = len(np.unique(combined[:,0]))
nb_movies = len(np.unique(combined[:,1]))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)


# convert to Torch sensors 

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#movies without ratings were initialy given value of 0, turn that no -1 since we want rating <= 2 to be 0 and >2 to be 1

training_set[training_set== 0] = -1
training_set[training_set== 1] = 0
training_set[training_set== 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set== 0] = -1
test_set[test_set== 1] = 0
test_set[test_set== 2] = 0
test_set[test_set >= 3] = 1 

# f = Variable(test_set[1-1]).unsqueeze(0)
# f = f.data.numpy()

### RBM Class

class RBM():
    def __init__(self,nv,nh):#visible nodes,hidden nodes
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)#hidden node bias, proability hiddem node equals one given a visible node
        self.b = torch.randn(1,nv)#visible node bias, proability visible node equals one given a hidden node
    def sampleh(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)#basically y =mx +c we u/ltiply the weight to the input n add the bias, this is used to calculate wheher the given value will trigger the activation function n hence fire off the node
        p_h_given_v = torch.sigmoid(activation)#passing ^ into activation function 
        return p_h_given_v  , torch.bernoulli(p_h_given_v) #applying bernoulli sampling
    def samplev(self,y):
        wy = torch.mm(y,self.W)#dont need to transpose 
        activation = wy + self.b.expand_as(wy)#basically y =mx +c we multiply the weight to the input n add the bias, this is used to calculate wheher the given value will trigger the activation function n hence fire off the node
        p_v_given_h= torch.sigmoid(activation)#passing ^ into activation function 
        return p_v_given_h  , torch.bernoulli(p_v_given_h) #applying bernoulli sampling
    def train(self,v0, vk, ph0, phk):#hadling contrastive divergence
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)  #diff between input vector of observations and the vector of observations of the visible nodes after k samplings.
        self.a += torch.sum((ph0 - phk), 0)#ph0,is the probability hidden node equal to 1 given values of v0, where v0 is the input vector of observations.phk is the probabilty the hidden node equal to 1 given values of vk where vk is the values of visible nodes after k iterations
    def predict( self, v): # v: visible nodes
        _, h = self.sampleh(v)
        _, v = self.samplev(h)
        return v

        
        
nv = len(training_set[0])
nh = 104
batch_size = 100    
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 16
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]#the values of the visible nodes after k iterations
        v0 = training_set[id_user:id_user+batch_size]#values of the visible nodes during the first input, source of truths
        ph0,_ = rbm.sampleh(v0)#initial probabilty of hdden node equal 1, given real ratings from source of truth
        for k in range(10):#k-step contrastive divergence w gibbs sampling
            _,hk = rbm.sampleh(vk)#hk are the hiden nodes obtained aft the kth step of contrastive dvergence,here we sample the kth hidden node,using bernoulli sampling we do this by calling def(sampleh) on the visible node that fed information to the said kth hidden node,this returns us the the sampled hidden nodes
            _,vk = rbm.samplev(hk)#vk are the visible nodes obtained aft kth step of contrastive divergence,here we sample the kth visible node by using bernoulli sampling. we do this by calling def(samplev) on the hidden node that fed information to the said kth visible node,this returns us the the sampled kth visible node. we then update the value of vk, dont update or apply sampling to v0,it is the source of truth
            vk[v0<0] = v0[v0<0]#we dont want to update values of cells which werent rated by the user
        phk,_ = rbm.sampleh(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))#we compare our predictions of the machine, values of visible nodes at kth iteraion against the values of visible node before Gibbs sampling, only for cells rated by users
        s += 1.#we add tto normalise the train_loss later on
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sampleh(v)
        _,v = rbm.samplev(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))#here, we use the model to predict values of movies not rated by users, which is v0[v0<0] or v[v<0]. In this case.and it is predicted based off of v, the input data,and then we compare it against vt, the test set data which contains the values for which the same user rated those movies that they didn't rate initially hence vt is the source of truth and v is the what we compare it against
        s += 1.
print('test loss: '+str(test_loss/s))

user_id = 1
user_input = Variable(test_set[user_id-1]).unsqueeze(0)
x=user_input.data.numpy()
output = rbm.predict(user_input)
output = output.data.numpy()
input_output = np.vstack([user_input, output])


