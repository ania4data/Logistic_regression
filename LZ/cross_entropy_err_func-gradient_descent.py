
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[95]:


N=100
D=2

X=np.random.randn(N,D)
X[:50,:]=X[:50,:]-2*np.ones((50,D))
X[50:,:]=X[50:,:]+2*np.ones((50,D))

T_target=np.random.randn(N)
T_target[:50]=0
T_target[50:]=1      # another way T=np.array([0]*50+[1]*50)    shape(100,)  array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
Xn=np.concatenate((np.ones((100,1)),X),axis=1)                  #if do N,1 is (100,1), Y_predict is (100,)

w=(np.random.randn(D+1))
z=np.dot(Xn,w)

def sigmoid(z):
    return 1/(1+np.exp(-z))


alpha=0.1    # learning rate

Y_n=sigmoid(z)

def cross_entropy_err(Yp,t):
    Err=(-1.0*(t*np.log(Yp)+(1.0-t)*np.log(1-Yp)))
    return np.sum(Err)

for time in range(1000):

    derivative=np.dot(Xn.T,(T_target-Y_n))
    w=w+alpha*derivative
    Y_n=sigmoid(np.dot(Xn,w))
    print(time,cross_entropy_err(Y_n,T_target))
    #print(w,time)
print(w,time)    


#plt.scatter(X[:,0],X[:,1],c=T_target,s=100,alpha=0.5)

#x_axis=np.linspace(-6,6,100)
#y_axis=-1.*np.linspace(-6,6,100)
#plt.plot(x_axis,y_axis)


# In[42]:


np.shape(derivative)

