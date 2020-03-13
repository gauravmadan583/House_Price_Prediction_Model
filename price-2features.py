import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def costFunction(theta,X,y):
    m=y.size
    J=(X@theta-y).transpose()@(X@theta-y)
    return J/(2*m)
def gradientdescent(theta,X,y,iterations):
    J_history=np.zeros(iterations)
    m=y.size
    alpha=0.1
    for iteration in range(iterations):
        theta=theta-(X.transpose()@(X@theta-y))*alpha/m
        J_history[iteration]=costFunction(theta,X,y)
    return theta,J_history
def normalize(X,y):
    theta=np.zeros((X.shape[1],1))
    theta=np.linalg.inv(X.transpose()@X)@X.transpose()@y
    return theta
def mean(X):
    # print(sum(X)/X.shape[0])
    return sum(X)/X.shape[0]
def featurise(Y):
    X=Y.transpose()
    for i in range(X.shape[0]):
        X[i]=X[i]-mean(X[i])
        X[i]=X[i]/(max(X[i])-min(X[i]))
    return X.transpose()
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
# print(x
X=np.concatenate((q,x),axis=1)
# print(X)
theta=np.zeros((X.shape[1],1))
theta,J_history=gradientdescent(theta,X,y,400)
# print(theta)
# plt.plot(J_history)
# plt.show()
Y=X@theta
print(x[:,:1])
plt.scatter(x[:,:1],y,color="blue")
plt.scatter(x[:,1:2],y,color="red")
# plt.plot(x[:,:1],Y)
plt.show()
# print(x[:,1:2])
# plt.plot(x[:,1:2],Y,color="red")
# plt.show()
