#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import numpy as np;\nimport matplotlib.pyplot as plt;\nimport pandas as pd;\nfrom sklearn.preprocessing import LabelEncoder, OneHotEncoder , StandardScaler;\nfrom sklearn.compose import ColumnTransformer;\nfrom sklearn.model_selection import train_test_split , GridSearchCV;\nfrom sklearn.metrics import confusion_matrix , accuracy_score\n\n!pip install --upgrade pip\n!pip install tensorflow\n\nimport tensorflow as tf;')


# In[ ]:


df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values#Independent Variables, the (.values) returns list where each item is a
#row(horizontal)in the form of a list thus creating a list of lists
Y = df.iloc[:, 13].values#Dependent Variable, the (.values) returns data in each
#row(horizontal) as an element in a list


# In[ ]:


#Encoding the Categorical Data , which are the Country and Gender
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])#label Encoding of Gender column
X[:,1] = le.fit_transform(X[:,1])#label Encoding of Geography column


# In[ ]:


columns = pd.read_csv('Churn_Modelling.csv').columns#get column names from our original dataset
columns_X = columns[3:13]#get column names of only the independent variables


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


X_df = pd.DataFrame(data=X, columns = columns_X)#display our independent variables aa a df
X_df.head(3)#show the first 3


# In[ ]:


#Split into training and test set
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)#random seed 42


# In[ ]:


x_train_df = pd.DataFrame(data=x_train, columns = columns_X)
x_train_df.columns
x_train_df = x_train_df[['Geography', 'Gender',  'HasCrCard', 'IsActiveMember','CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts','EstimatedSalary']]


x_test_df = pd.DataFrame(data=x_test, columns = columns_X)
x_test_df.columns
x_test_df = x_test_df[['Geography', 'Gender',  'HasCrCard', 'IsActiveMember','CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts','EstimatedSalary']]


# In[ ]:


x_train_df.head(5)


# In[ ]:


# We shall now standardise our Independent Variables
sc = StandardScaler()


x_train_df=  sc.fit_transform(x_train_df)
x_test_df =  sc.transform(x_test_df)


# In[ ]:


#Initialising ANN
ann = tf.keras.models.Sequential()


# In[ ]:


ann.add(tf.keras.layers.Dense(6 ,activation='relu'))#2 hidden layers with 6 nodes each
tf.keras.layers.Dropout(0.2)
ann.add(tf.keras.layers.Dense(6 ,activation='relu'))
tf.keras.layers.Dropout(0.1)


# In[ ]:


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[ ]:


ann.compile(optimizer='Adam' , loss='binary_crossentropy' , metrics=['accuracy','RootMeanSquaredError'])


# In[ ]:


ann.fit(x_train_df, y_train ,batch_size=64 , epochs=100)


# In[ ]:


y_pred = ann.predict(x_test_df)
y_pred = np.where(y_pred>=0.5,1, y_pred )
y_pred = np.where(y_pred < 0.5,0, y_pred )
y_pred = tf.squeeze(y_pred, axis=1)
y_pred = np.array(y_pred)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)
ann.evaluate(x_test_df , y_test)
cm


# In[ ]:





# In[ ]:





# In[ ]:


#K-FOLD Cross Validation


# In[ ]:


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


def build_model():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(12, input_dim=10, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[ ]:


X= pd.DataFrame(data=X, columns = columns_X)
# We shall now standardise our Independent Variables
sc = StandardScaler()
x_train_df=  sc.fit_transform(x_train_df)


# In[ ]:


X= X[['Geography', 'Gender',  'HasCrCard', 'IsActiveMember','CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts','EstimatedSalary']]


# In[ ]:


model = KerasClassifier(build_fn=build_model, epochs=150, batch_size=32, verbose=0)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:


# Tunning ANN with gridsearch


# In[ ]:


parameters = {"batch_size":[53,64,100] ,
              "epochs":[75,100,150],


                }

grid_search=GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring ='accuracy' , 
                          cv=10 )


# In[ ]:


grid_search = grid_search.fit(x_train_df , y_train)


# In[ ]:


best_param = grid_search.best_params_


# In[ ]:


best_accuracy = grid_search.best_score_


# In[ ]:


best_param


# In[ ]:


best_accuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




