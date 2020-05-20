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
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
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




# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


#training the model
         
sae = SAE()
criterion = nn.MSELoss()#criterion for the loss funciton,mean square error
optimizer = optim.RMSprop(sae.parameters(),lr= 0.01, weight_decay = 0.5 )

nb_epoch = 200

for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) >0 :
            output = sae(input)#forward function applied
            target.require_grad = False#ensure that we dont apply stochastic gradient descent to target, only the input
            output[target == 0] = 0 #we are setting movies that werent rated by the user to 0,therefore saving time and memory as we needent count them in the computations of the error so they arent included in updating weights
            loss = criterion(output,target)#compare output vs real to get loss
            mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1 
            optimizer.step()
    print('epoch: ' +str(epoch) + 'loss: ' + str(train_loss/s))
    
    
    
    
    
##testing model

    

for epoch in range(1,nb_epoch+1):
    test_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = Variable(test_set[id_user]).unsqueeze(0)
        if torch.sum(target.data > 0) >0 :
            output = sae(input)#forward function applied
            target.require_grad = False#ensure that we dont apply stochastic gradient descent to target, only the input
            output[target == 0] = 0 #we are setting movies that werent rated by the user to 0,therefore saving time and memory as we needent count them in the computations of the error so they arent included in updating weights
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10)
            # loss.backward()#removed as we dont need this,only for trg
            test_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1 
            optimizer.step()
    print('epoch: ' +str(epoch) + 'loss: ' + str(test_loss/s))
    
    
    
    
     
    
    
    
    