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

#split the data into training set and the testing set
def train_test_split(X,Y,test_size):
    X_train=X[:math.floor(len(X)*(1-test_size))]
    Y_train=Y[:math.floor(len(Y)*(1-test_size))]
    X_test=X[math.floor(len(X)*(1-test_size)):]
    Y_test=Y[math.floor(len(Y)*(1-test_size)):]
    return X_train, X_test, Y_train, Y_test

#hypothesis function(M=2):
def hypothesis(theta,X):
    value=0
    for i in range(0,18):
        value+=theta[i]*X[i]
    for l in range(18,171):
        for j in range(1,18):
            for k in range(1,j+1):
                value+=theta[l]*X[j]*X[k]
    return value

#calculate the cost function
def square_error(theta,X,T):
    sqr_error=0
    for i in range(0,len(X)):
        sqr_error+=(hypothesis(theta,X[i])-T[i])**2
    return sqr_error/len(X)/2

#gradient descent
def gradient_descent(X,T,theta,learning_rate,iteration):
    N=len(X)
    cost_function=[]
    for i in range(iteration):
        theta_grad=[0]*len(theta)
        for j in range(0,N):
            theta_grad[0]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,0]
            theta_grad[1]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,1]
            theta_grad[2]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,2]
            theta_grad[3]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,3]
            theta_grad[4]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,4]
            theta_grad[5]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]
            theta_grad[6]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]
            theta_grad[7]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]
            theta_grad[8]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]
            theta_grad[9]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]
            theta_grad[10]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]
            theta_grad[11]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]
            theta_grad[12]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]
            theta_grad[13]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]
            theta_grad[14]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]
            theta_grad[15]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]
            theta_grad[16]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]
            theta_grad[17]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]

            theta_grad[18]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,1]*X[j,1]
            theta_grad[19]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,2]*X[j,1]
            theta_grad[20]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,2]*X[j,2]
            theta_grad[21]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,3]*X[j,1]
            theta_grad[22]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,3]*X[j,2]
            theta_grad[23]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,3]*X[j,3]
            theta_grad[24]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,4]*X[j,1]
            theta_grad[25]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,4]*X[j,2]
            theta_grad[26]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,4]*X[j,3]
            theta_grad[27]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,4]*X[j,4]
            theta_grad[28]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]*X[j,1]
            theta_grad[29]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]*X[j,2]
            theta_grad[30]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]*X[j,3]
            theta_grad[31]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]*X[j,4]
            theta_grad[32]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,5]*X[j,5]
            theta_grad[33]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,1]
            theta_grad[34]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,2]
            theta_grad[35]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,3]
            theta_grad[36]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,4]
            theta_grad[37]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,5]
            theta_grad[38]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,6]*X[j,6]
            theta_grad[39]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,1]
            theta_grad[40]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,2]
            theta_grad[41]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,3]
            theta_grad[42]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,4]
            theta_grad[43]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,5]
            theta_grad[44]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,6]
            theta_grad[45]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,7]*X[j,7]
            theta_grad[46]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,1]
            theta_grad[47]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,2]
            theta_grad[48]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,3]
            theta_grad[49]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,4]
            theta_grad[50]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,5]
            theta_grad[51]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,6]
            theta_grad[52]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,7]
            theta_grad[53]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,8]*X[j,8]
            theta_grad[54]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,1]
            theta_grad[55]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,2]
            theta_grad[56]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,3]
            theta_grad[57]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,4]
            theta_grad[58]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,5]
            theta_grad[59]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,6]
            theta_grad[60]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,7]
            theta_grad[61]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,8]
            theta_grad[62]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,9]*X[j,9]
            theta_grad[63]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,1]
            theta_grad[64]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,2]
            theta_grad[65]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,3]
            theta_grad[66]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,4]
            theta_grad[67]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,5]
            theta_grad[68]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,6]
            theta_grad[69]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,7]
            theta_grad[70]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,8]
            theta_grad[71]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,9]
            theta_grad[72]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,10]*X[j,10]
            theta_grad[73]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,1]
            theta_grad[74]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,2]
            theta_grad[75]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,3]
            theta_grad[76]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,4]
            theta_grad[77]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,5]
            theta_grad[78]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,6]
            theta_grad[79]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,7]
            theta_grad[80]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,8]
            theta_grad[81]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,9]
            theta_grad[82]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,10]
            theta_grad[83]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,11]*X[j,11]
            theta_grad[84]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,1]
            theta_grad[85]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,2]
            theta_grad[86]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,3]
            theta_grad[87]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,4]
            theta_grad[88]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,5]
            theta_grad[89]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,6]
            theta_grad[90]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,7]
            theta_grad[91]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,8]
            theta_grad[92]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,9]
            theta_grad[93]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,10]
            theta_grad[94]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,11]
            theta_grad[95]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,12]*X[j,12]
            theta_grad[96]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,1]
            theta_grad[97]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,2]
            theta_grad[98]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,3]
            theta_grad[99]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,4]
            theta_grad[100]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,5]
            theta_grad[101]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,6]
            theta_grad[102]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,7]
            theta_grad[103]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,8]
            theta_grad[104]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,9]
            theta_grad[105]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,10]
            theta_grad[106]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,11]
            theta_grad[107]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,12]
            theta_grad[108]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,13]*X[j,13]
            theta_grad[109]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,1]
            theta_grad[110]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,2]
            theta_grad[111]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,3]
            theta_grad[112]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,4]
            theta_grad[113]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,5]
            theta_grad[114]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,6]
            theta_grad[115]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,7]
            theta_grad[116]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,8]
            theta_grad[117]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,9]
            theta_grad[118]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,10]
            theta_grad[119]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,11]
            theta_grad[120]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,12]
            theta_grad[121]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,13]
            theta_grad[122]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,14]*X[j,14]
            theta_grad[123]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,1]
            theta_grad[124]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,2]
            theta_grad[125]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,3]
            theta_grad[126]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,4]
            theta_grad[127]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,5]
            theta_grad[128]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,6]
            theta_grad[129]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,7]
            theta_grad[130]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,8]
            theta_grad[131]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,9]
            theta_grad[132]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,10]
            theta_grad[133]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,11]
            theta_grad[134]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,12]
            theta_grad[135]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,13]
            theta_grad[136]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,14]
            theta_grad[137]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,15]*X[j,15]
            theta_grad[138]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,1]
            theta_grad[139]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,2]
            theta_grad[140]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,3]
            theta_grad[141]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,4]
            theta_grad[142]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,5]
            theta_grad[143]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,6]
            theta_grad[144]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,7]
            theta_grad[145]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,8]
            theta_grad[146]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,9]
            theta_grad[147]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,10]
            theta_grad[148]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,11]
            theta_grad[149]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,12]
            theta_grad[150]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,13]
            theta_grad[151]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,14]
            theta_grad[152]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,15]
            theta_grad[153]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,16]*X[j,16]
            theta_grad[154]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,1]
            theta_grad[155]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,2]
            theta_grad[156]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,3]
            theta_grad[157]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,4]
            theta_grad[158]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,5]
            theta_grad[159]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,6]
            theta_grad[160]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,7]
            theta_grad[161]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,8]
            theta_grad[162]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,9]
            theta_grad[163]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,10]
            theta_grad[164]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,11]
            theta_grad[165]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,12]
            theta_grad[166]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,13]
            theta_grad[167]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,14]
            theta_grad[168]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,15]
            theta_grad[169]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,16]
            theta_grad[170]+=(1/N)*(hypothesis(theta,X[j])-T[j])*X[j,17]*X[j,17]
        for l in range(0,len(theta)):
            theta[l]-=learning_rate*theta_grad[l]
        cost_function.append(square_error(theta,X_train,T_train))
    return theta,cost_function

#rmse
def rmse(y1,y2):
    error=0
    for i in range(len(y1)):
        error+=(y1[i]-y2[i])**2
    return math.sqrt(error/len(y1))

#setting the parameter
learning_rate=0.00001
theta=[0]*171
iteration=1

#split the dataset into training dataset and testing dataset
X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.1)
#train:876 test:220
'''for i in range(0,len(X_train)):
    print(hypothesis(theta,X_train[i]))
end_time=time.time()
print("the runtime of this program:%.3lf" %(end_time-start_time))'''

#the initial weight value and the final weight value
print("Initial state:")
print("theta=",theta)
print("Running for the method of gradient descent")
theta,cost_function=gradient_descent(X_train,T_train,theta,learning_rate,iteration)
print("Final state:")
print("theta=",theta)

#plot the cost function versus iteration times
for i in range(0,len(cost_function)):
    plt.plot(i,cost_function[i],'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=[]
for i in range(0,len(X_train)):
    y.append(hypothesis(theta,X_train[i]))
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=1) training")
plt.legend()
plt.show()

#plot the value of the model predict and the actual model (test part)
x=np.arange(0,len(X_test))
y=[]
for i in range(0,len(X_test)):
    y.append(hypothesis(theta,X_test[i]))
plt.plot(x,y,color='red',lw=1.0,ls='-',label="testing_predict_value")
plt.plot(x,T_test,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_test,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=1) testing")
plt.legend()
plt.show()

#the runtime of the program
end_time=time.time()
print("the runtime of this program:%.3lf" %(end_time-start_time))