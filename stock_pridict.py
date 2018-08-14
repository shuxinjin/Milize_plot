# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:17:51 2018

@author: sxjin

"""


#exported the scraped stock data from  a csv file. 
#The dataset contains n = 41266 minutes of data ranging from April to August
#2017 on 500 stocks as well as the total S&P 500 index price. 
#Index and stocks are arranged in wide format.



# Import
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data from csv file

data = pd.read_csv('sp500/data_stocks.csv')



# Drop date variable
# process the date colum. if 1 means column, if 0 means row .
#there are some special functions in pandas,if you need process such data format
#,you need learn the spection functions with it.
data = data.drop(['DATE'], 1)



# Dimensions of dataset
#matrix have one attribute ,column and row .Use shape() method to show the number.
n = data.shape[0]

p = data.shape[1]



# Make data a np.array

data = data.values



# Training and test data

train_start = 0
# 80% data as the test dataset. 
# floor() returns the largest integer not greater than the input parameter
train_end = int(np.floor(0.8*n))

#test dataset
test_start = train_end + 1

test_end = n


#slice operation,
data_train = data[np.arange(train_start, train_end), :]

#target to return a ndarray type ,then slice them 
data_test = data[np.arange(test_start, test_end), :]



# Scale data  
# Most NN data need data normalization , because the data range of tanh and sigm characteristic
# transfer dataset between range of min and max
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler.fit(data_train)

#calculate as scaler function defination.
data_train = scaler.transform(data_train)

data_test = scaler.transform(data_test)



# Build X and y
#1 is the index of column, 1 to last column 
X_train = data_train[:, 1:]
#0 column data, this column is the Y target data
y_train = data_train[:, 0]

#test
X_test = data_test[:, 1:]

y_test = data_test[:, 0]



# Number of stocks in training data
#shape 0 is matrix row number ,shape 1 is the column number .
n_stocks = X_train.shape[1]



# Neurons
#be used in below formula

n_neurons_1 = 1024

n_neurons_2 = 512

n_neurons_3 = 256

n_neurons_4 = 128



# Session
# apply interactive session in current project.
net = tf.InteractiveSession()



# Placeholder. place holder is the theroy .deployed the place holder

X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])

Y = tf.placeholder(dtype=tf.float32, shape=[None])



# Initializers

sigma = 1

#there are several kind of initializer type, this is one of them .
#a rd article ,https://blog.csdn.net/m0_37167788/article/details/79073070
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

bias_initializer = tf.zeros_initializer()



# Hidden weights

W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))

bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))

bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))

bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))

bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))



# Output weights
# calculate the output.

W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))

bias_out = tf.Variable(bias_initializer([1]))



# Hidden layer .

#1. linear transformation for neural unit.
#2. tf.matmul(..., ...),Multiply the matrix, the number of left matrix 
#   columns is equal to the number of right matrix rows.
#3. tf.add(..., ...) —— addition
#4. np.random.randn(...)——  random init

hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))



# Output layer (transpose!)
# axis transpose. one article help you understand this ,
# https://blog.csdn.net/lothakim/article/details/79494782
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))



# Cost function

#The loss function of the network is mainly used to generate a deviation value 
#between the network prediction and the actually observed training target.
#For regression problems, the mean square error (MSE) function is most commonly used. 
#The MSE calculates the average squared error between the predicted and target values.

mse = tf.reduce_mean(tf.squared_difference(out, Y))




'''
The optimizer handles the necessary calculations for adapting to network weights 
and bias variables during training. These calculations call the gradient 
calculation results, indicating the direction in which the weights and 
deviations need to be changed during training, thereby minimizing the cost 
function of the network. The development of stable and fast optimizers has 
always been an important research in the field of neural networks and deep learning.



The above is the use of the Adam optimizer, which is the default optimizer 
in deep learning today. Adam stands for Adaptive Moment Estimation and can
 be used as a combination of the two optimizers AdaGrad and RMSProp.
'''
# Optimizer

opt = tf.train.AdamOptimizer().minimize(mse)





'''
The initializer is used to initialize the variables of the network before 
training. Because neural networks are trained using numerical optimization 
techniques, the starting point for optimization problems is the focus of 
finding a good solution. There are different initializers in TensorFlow,
 each with a different initialization method. In this article, I am using tf.
 variance_scaling_initializer(), which is a default initialization strategy.
'''
# Init

net.run(tf.global_variables_initializer())



# Setup plot

plt.ion()

fig = plt.figure()

ax1 = fig.add_subplot(111)

line1, = ax1.plot(y_test)

line2, = ax1.plot(y_test * 0.5)

plt.show()



# Fit neural net

batch_size = 256

mse_train = []

mse_test = []



# Run

epochs = 10

for e in range(epochs):



    # Shuffle training data

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    X_train = X_train[shuffle_indices]

    y_train = y_train[shuffle_indices]



    # Minibatch training

    for i in range(0, len(y_train) // batch_size):

        start = i * batch_size
        #slice
        batch_x = X_train[start:start + batch_size]

        batch_y = y_train[start:start + batch_size]

        # Run optimizer with batch
        # run the session 
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})



         # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(y_test)
            
            ax1.plot(pred.transpose())
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            
            plt.pause(0.01)  