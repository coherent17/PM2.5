#M=1
import numpy as np 
import matplotlib.pyplot as plt 
import math

dataT=np.genfromtxt('data_T.csv',delimiter=',')
dataX=np.genfromtxt('data_X.csv',delimiter=',')

def train_test_split(X,Y,test_size):
    X_train=X[:math.floor(len(X)*(1-test_size))]
    Y_train=Y[:math.floor(len(Y)*(1-test_size))]
    X_test=X[math.floor(len(X)*(1-test_size)):]
    Y_test=Y[math.floor(len(Y)*(1-test_size)):]
    return X_train, X_test, Y_train, Y_test

def hypothesis(theta,X):
    value=0
    for i in range(0,len(theta)):
        value+=theta[i]*X[i]
    return value

def square_error(theta,X,T):
    sqr_error=0
    for i in range(0,len(X)):
        sqr_error+=(hypothesis(theta,X[i])-T[i])**2
    return sqr_error/len(X)/2

def gradient_descent(X,T,theta,learning_rate,iteration):
    N=len(X)
    cost_function=[]
    for i in range(iteration):
        theta_grad=[0]*len(X[0])
        for j in range(0,N):
            for k in range(0,len(X[0])):
                theta_grad[k]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,k]
            for k in range(0,len(X[0])):
                theta[k]-=learning_rate*theta_grad[k]
        cost_function.append(square_error(theta,X_train,T_train))
    return theta,cost_function

#rmse
def predict_error(theta,X,T):
    error=0
    for i in range(0,len(X)):
        error+=(hypothesis(theta,X[j])-T[j])**2
    return error/len(X)

learning_rate=0.0000001
theta=[0]*18
iteration=500

X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.2)
#train:876 test:220



print("Initial state:")
print("theta=",theta)
print("Running for the method of gradient descent")
theta,cost_function=gradient_descent(X_train,T_train,theta,learning_rate,iteration)
print("Final state:")
print("theta=",theta)


for i in range(0,len(cost_function)):
    plt.plot(i,cost_function[i],'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()

#plot the value of the model predict and the actual model
for i in range(0,len(X_test)):
    plt.plot(i,T_test[i],'r_')
    plt.plot(i,hypothesis(theta,X_test[i]),'y_')
plt.show()