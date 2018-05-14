
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_data():


    df=pd.read_csv('ecommerce_data.csv')
    data=df.as_matrix()

    X=data[:,:-1]
    Y=data[:,-1]
    X[:,1]=(X[:,1]-X[:,1].mean())/(X[:,1].std())   #number of product viewed  0-4
    X[:,2]=(X[:,2]-X[:,2].mean())/(X[:,2].std())   #visit duration 0-6.3

    N,D=X.shape
    X2=np.zeros((N,4))
    for item in range(N):

        if (int(X[item,4])==0):             #anotherway   Z=np.zeros((N,4))  Z[np.arrange(N),X[:,D-1].astype(np.int32)]=1
            X2[item,0]=1                    #another way xreate X2, file in X to X2 from 0, D-1, X2[:,0:(D-1)]=X[:,0:(D-1)]
        if (int(X[item,4])==1):              #do a for loop item in range(N), X2[item,int(X[item,D-1])]=1
            X2[item,1]=1
        if (int(X[item,4])==2):
            X2[item,2]=1
        if (int(X[item,4])==3):
            X2[item,3]=1

    Xn=np.concatenate((X,X2),axis=1)
    return Xn,Y

def get_binary_data():

    X,Y=get_data()
    X2=X[Y<=1]
    Y2=Y[Y<=1]
    return X2,Y2
