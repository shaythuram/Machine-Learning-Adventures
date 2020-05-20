

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# # Importing the dataset
# dataset = pd.read_csv('Churn_Modelling.csv')
# X = dataset.iloc[:, 3:13].values
# y = dataset.iloc[:, 13].values

# Encoding categorical data, we encoding the country via x_1 and x_2 for gender, then remove 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Do onehotencoding for for the countries since we have 0,1,2 so we remove one to avoid dummy variable 
#trap not needed for gender since only 0 and 1
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#Making the ANN

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout

###########################################moved into the build function ########################

classifier = Sequential()

#1st hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
# classifier.add(Dropout())

#2nd hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))

#ouput layer
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

#Compilingthe ANN,settling the weights, use logarathimic loss for binary output
classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

###########################################moved into the build function ########################

 
#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10, epochs=100)
 

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)#sets that allvallues of y>0.5 returns true

#gen confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    #1st hidden layer
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
    #2nd hidden layer
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
    #ouput layer
    classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
    #Compiling the ANN,settling the weights, use logarathimic loss for binary output
    classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])    
    #fitting the ANN to the training set
    return classifier
classifier = KerasClassifier(build_fn=build_classifier , batch_size=10 , epochs=100)

#we are gonna use k-fold validation, in our batch of 10, we run 9 lines of training and one line of testing, 

accuracies = cross_val_score(estimator = classifier ,X = X_train , y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()



# Tuning our ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    #1st hidden layer
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
    #2nd hidden layer
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
    #ouput layer
    classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
    #Compilingthe ANN,settling the weights, use logarathimic loss for binary output
    classifier.compile(optimizer=optimizer , loss='binary_crossentropy' , metrics=['accuracy'])    
    #fitting the ANN to the training set
    return classifier
classifier = KerasClassifier(build_fn=build_classifier , batch_size=10 , epochs=100)

parameters = {"batch_size":[25,32],
              "epochs" : [100,500],
              "optimizer" :['adam' , 'rmsprop'] }

grid_search=GridSearchCV(estimator = classifier, 
                          param_grid = parameters,
                          scoring ='accuracy' , 
                          cv=10 )

grid_search = grid_search.fit(X_train,y_train)
  
best_param = grid_search.best_params_
  
best_accuracy = grid_search.best_score_

