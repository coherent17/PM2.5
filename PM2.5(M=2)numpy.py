#M=2
import numpy as np 
import matplotlib.pyplot as plt 
import math
import time

#calculate the runtime of the program
start_time = time.time()

#read the data which X features has add a new line 1 so means X_0
dataT=np.genfromtxt('data_T.csv',delimiter=',')
dataX=np.genfromtxt('data_X.csv',delimiter=',')

#append the dataX to match the theta (171 features)
k=18
for i in range(1,18):
    for j in range(1,i+1):
        if k in range(18,171):
            dataX=np.insert(dataX,k,values=dataX[:,i]*dataX[:,j],axis=1)
            k+=1

#split the data into training set and the testing set
def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    Y_train=Y_train.reshape(1,len(Y_train))
    Y_test=Y_test.reshape(1,len(Y_test))
    return X_train, X_test, Y_train, Y_test

#hypothesis function
def hypothesis(theta,X):
    return np.matmul(theta,np.transpose(X))

#gradient descent
def gradient_descent(theta,X,T,learning_rate,iteration):
    N=len(X)
    cost_function=[]
    for i in range(1,iteration+1):
        if i ==1:
            print("before iteration, rmse is %.8lf and cost function is %.8lf" %(rmse(hypothesis(theta,X).reshape(len(X),),T),np.sum((hypothesis(theta,X)-T)**2)/len(X)/2))
        cost_function.append(np.sum((hypothesis(theta,X)-T)**2)/len(X)/2)
        theta_grad=(1/N)*np.matmul((hypothesis(theta,X)-T),(X))
        theta-=learning_rate*theta_grad
        if i %(iteration/10)==0:
            print("it is the %d time of iterations, rmse is %.8lf and cost function is %.8lf" %(i,rmse(hypothesis(theta,X).reshape(len(X),),T),np.sum((hypothesis(theta,X)-T)**2)/len(X)/2))
    return theta,cost_function

#root mean square error
def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

#parameter:
learning_rate=0.00000000003
iteration=50000
theta=np.zeros((1,171))

#split the data into training set and the testing set
X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.2)

#the initial weight value and the final weight value
print("Initial state:")
print("theta=",theta)
print("Running for the method of gradient descent")
theta,cost_function=gradient_descent(theta,X_train,T_train,learning_rate,iteration)
print("Final state:")
print("theta=",theta)

end_time=time.time()
print("the runtime of this program:%.3lf" %(end_time-start_time))

#plot the cost function versus iteration times
x=np.arange(0,len(cost_function))
plt.plot(x,cost_function,'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=hypothesis(theta,X_train).reshape(len(X_train),)
T_train=T_train.reshape(len(X_train),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=2) training")
plt.legend()
plt.show()

#plot the value of the model predict and the actual model (test part)
x=np.arange(0,len(X_test))
y=hypothesis(theta,X_test).reshape(len(X_test),)
T_test=T_test.reshape(len(X_test),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="testing_predict_value")
plt.plot(x,T_test,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_test,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=2) testing")
plt.legend()
plt.show()