#Cost Function
def costFunction(theta,X,y):
    m=y.shape[0]
    J=((X@theta-y).transpose())@(X@theta-y)
    return J/(2*m)
#Gradient Descent- It will return theta after the no. of iterations and J_history
def gradientdescent(theta,X,y,iterations):
    alpha=0.01
    m=y.shape[0]
    J_history=np.zeros((iterations))
    for i in range(iterations):
        theta=theta-(X.transpose())@(X@theta-y)*alpha/m
        J_history[i]=costFunction(theta,X,y)
    return theta,J_history
#Feature Normalisation of Data to obtain the minima faster.
def normalize(X,y):
    theta=np.zeros((X.shape[1],1))
    theta=np.linalg.inv(X.transpose()@X)@X.transpose()@y
    return theta
f=open("ex1data1.txt","r")
f1=f.readlines()
count = 0
for line in f1: count += 1
import numpy as np
x=np.ones((count,1))
Y=np.ones((count,1))
for i in range(count) :
    x[i][0],Y[i][0]=f1[i].split(",")
q=np.ones((count,1))
X=np.concatenate((q,x),axis=1)
theta=np.zeros((X.shape[1],1))
theta,J_history=gradientdescent(theta,X,Y,1500)
print(theta)
print(normalize(X,Y))
import matplotlib.pyplot as plt
plt.scatter(x,Y)
y=X@theta
plt.plot(x,y,color="red")
plt.show()
plt.plot(J_history)
plt.show()
