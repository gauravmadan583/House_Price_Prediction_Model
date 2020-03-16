import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def costFunction(theta,X,y):
    m=y.shape[0]
    J=((X@theta-y).transpose())@(X@theta-y)
    return J/(2*m)

def gradientdescent(theta,X,y,iterations):
    J_history=np.zeros(iterations)
    m=y.shape[0]
    alpha=0.1
    for iteration in range(iterations):
        theta=theta-(X.transpose()@(X@theta-y))*alpha/m
        J_history[iteration]=costFunction(theta,X,y)

    return theta,J_history

def normalEquation(X,y):
    theta=np.zeros((X.shape[1],1))
    theta=np.linalg.inv(X.transpose()@X)@(X.transpose())@y
    return theta

def mean(X):
    # print(sum(X)/X.shape[0])
    return sum(X)/X.size

def featurise(Y):
    mu=np.zeros((1,Y.shape[1]))
    diff=np.zeros((1,Y.shape[1]))
    Y_norm=Y
    for i in range(Y.shape[1]):
        mu[0][i]=mean(Y[:,i:i+1])
        diff[0][i]=max(Y[:,i:i+1])-min(Y[:,i:i+1])
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y_norm[i][j]=(Y[i][j]-mu[0][j])/diff[0][j]


    return Y_norm

f=open("ex1data2.txt","r")
f1=f.readlines()
count=0
for i in f1:
    count+=1
x=np.zeros((count,2),dtype="float64")
y=np.zeros((count,1),dtype="int64")
for i in range(count):
    a,b,c=f1[i].split(",")
    x[i][0]=float(a)
    x[i][1]=float(b)
    y[i][0]=float(c)

q=np.ones((count,1),dtype="float64")
x=featurise(x)
X=np.concatenate((q,x),axis=1)

theta=np.zeros((X.shape[1],1))
theta,J_history=gradientdescent(theta,X,y,100)

print(theta)

print(normalEquation(X,y))

plt.plot(J_history)
plt.show()
