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

def gradient_descent(theta,X,T,iteration):
    N=len(X)
    for i in range(iteration):
        theta_grad=np.zeros((1,N))
        for j in range(0,N):
            theta_grad[]+=(1/N)*(hypothesis(theta,X)-T)*X[j,:]
    return theta

X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.2)
T_train=T_train.reshape(1,876)
theta=np.ones((1,18))


temp=gradient_descent(theta,X_train,T_train,1)
# print(np.shape(theta))
# print(np.shape(T_train))
# print(np.shape(X_train))
# print(np.shape(hypothesis(theta,X_train)))
# print(temp)
# print(np.shape(temp))
# print(len(X_train))