#M=1
import numpy as np 
import matplotlib.pyplot as plt 
import math
import time

#calculate the runtime of the program
start_time = time.time()

#read the data which X features has add a new line 1 so means X_0
dataT=np.genfromtxt('data_T.csv',delimiter=',')
dataX=np.genfromtxt('data_X.csv',delimiter=',')

#split the data into training set and the testing set
def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    return X_train, X_test, Y_train, Y_test

#hypothesis function(M=1):
def hypothesis(theta,X):
    return np.matmul(theta,np.transpose(X))

def gradient_descent(theta,X,T,learning_rate,iteration):
    N=len(X)
    for i in range(iteration):
        theta_grad=(1/N)*np.dot((hypothesis(theta,X)-T),(X))
        theta-=learning_rate*theta_grad
    return theta

X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.2)
T_train=T_train.reshape(1,876)
theta=np.ones((1,18))

temp=gradient_descent(theta,X_train,T_train,0.00001,100)
print(temp)
 #theta:[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]] shape: (1, 18)
 #X_train shape: (876, 18)
 #np.transpose(X_train) shape: (18, 876)
 #hypothesis(theta,X_train) shape: (1, 876)