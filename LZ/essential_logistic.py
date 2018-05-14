
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.utils import shuffle






#
# 
#    def initilize_restart(epsilon=1e-10,iteration=100000,alpha=0.000001,lambda2_=2.0,lambda1_=0.0,w,w_mag,j_cost):
#    pass
#    return epsilon,iteration,w_mag,j_cost,alpha,lambda2_,lambda1_,w

#def initilize_basic(epsilon=1e-10,iteration=100000,alpha=0.000001,lambda2_=2.0,lambda1_=0.0,w,w_mag=[],j_cost=[]):
#    #epsilon=a,iteration=b,alpha=c,lambda2_=d,lambda1_=e
#    #epsilon=1e-10
#    #iteration=10000
#    #w_mag=[]
#    #j_cost=[]
#    
#    #alpha=0.001    # learning rate
#    #lambda2_=1
#    #lambda1_=0.01
#    pass
#    return epsilon,iteration,w_mag,j_cost,alpha,lambda2_,lambda1_,w


#---------------------------------------------
#Xtrain=np.random.randn(1400,D)
#Xtest=np.random.randn(600,D)


#Xtrain=X[:-600,:]
#Ytrain=Y[:-600]

#Xtest=X[-600:,:]
#Ytest=Y[-600:]

def shuffle_divide(X,Y,ratio):   #shuffle data with a ration, seed not specified, every time new data



    ratio=ratio
    X,Y=shuffle(X,Y)
    N=np.shape(X)[0]
    D=np.shape(X)[1]
    n_test=int(N*ratio)
    Xtrain=np.random.randn(N-n_test,D)
    Xtest=np.random.randn(n_test,D)
    
    Xtrain=X[0:N-n_test,:]
    Ytrain=Y[0:N-n_test]

    Xtest=X[N-n_test:N,:]
    Ytest=Y[N-n_test:N]
    return Xtrain,Ytrain,Xtest,Ytest
    
    
    
#---------------------------------------------    
def z_cal(Xn,w):
    return np.dot(Xn,w)
#---------------------------------------------
def sigmoid(z):
    return 1/(1+np.exp(-z))
#---------------------------------------------
def y_cal_z(Xn,w):
    z=z_cal(Xn,w)
    Y_n=sigmoid(z)
    return Y_n                  #Y from sigmoid of X.w
#---------------------------------------------    
def cross_entropy_err(Yp,t,epsilon):
    
    Err=(-1.0*(t*np.log(Yp+epsilon)+(1.0-t)*np.log(1-Yp+epsilon)))
    return np.sum(Err)           #j_cost log only

#---------------------------------------------

def w_initial(D):
    w_ini=(np.random.randn(D))
    return w_ini   #for problem with random initialization, not restarting

def get_restart_values(w,w_mag,j_cost) :
    pass
    return w,w_mag,j_cost


def gradient_descent(Xn,T_target,flag_restart,j_cost_restart,w_mag_restart,w_restart):


    D=np.shape(Xn)[1]
    
    if (flag_restart==0):   #start from random w

        epsilon=1e-10
        iteration=100000
        alpha=0.000001
        lambda2_=2.0
        lambda1_=0.0
        w=w_initial(D)
        w_mag=[]
        j_cost=[]
        #w,w_mag,j_
        #epsilon,iteration,w_mag,j_cost,alpha,lambda2_,lambda1_,w=initilize_basic()

    if (flag_restart==1):   #start from restart values here can read from restart files

        #f= open("restart_wmag2.txt","w")

        #for i in range(len(w_mag_restart)):
        #f.write(str(w_mag[i])+'\n')
        #f.close()

        #g= open("restart_w2.txt","w")

        #for i in range(len(w)):
        # g.write(str(w[i])+'\n')
        #g.close()

        #h= open("restart_j_cost2.txt","w")

        #for i in range(len(j_cost)):
        #h.write(str(j_cost[i])+'\n')
        #h.close() 
        epsilon=1e-10
        iteration=100000
        alpha=0.000001
        lambda2_=2.0
        lambda1_=0.0
        w=w_restart
        w_mag=w_mag_restart
        j_cost=j_cost_restart
        #w,w_mag,j_cost=get_restart_values()
        #epsilon,iteration,w_mag,j_cost,alpha,lambda2_,lambda1_,w=initilize_restart(epsilon=1e-10,iteration=100000,alpha=0.000001,lambda2_=2.0,lambda1_=0.0,w,w_mag,j_cost)
    else:
        print('need flag_restart: initialization 0 or restart 1')
  



    
    Y_n=y_cal_z(Xn,w)

    for time in range(iteration):
    

        derivative=np.dot(Xn.T,(T_target-Y_n))-lambda2_*w-lambda1_*(np.sign(w))            #keep the weights from overgrowing to make maximum liklihood maximum

        w=w+alpha*derivative 

        Y_n=y_cal_z(Xn,w)

        if (time%1==0):   #here can write to restart file as well every e.g. 1000 iteration, better to incase have two sets of restart incase crash happen
            
            j_cost.append(cross_entropy_err(Y_n,T_target,epsilon))
            w_mag.append(np.dot(w.T,w))

    return j_cost,w_mag,w,Y_n            
            
        
#---------------------------------------------          
def plotting(j_cost,w_mag):
    plt.plot(j_cost,label='lambda= 0.0')
    plt.show()
    plt.plot(w_mag,label='w_mag')
    plt.show()
    
#----------------------------------------------

def accuracy_rate(Y,T):    #accuracy
    
    N=len(T)
    print('Assume 1 is True for Actual data, False is 0')
    print('Classificatio rate :', 1-np.abs(np.round(T)-np.round(Y)).sum()/N)
    print('----------------------------------------------------------')

    
#    TP FP
#    FN TN
    
    c_tp=0
    c_fp=0
    c_tn=0
    c_fn=0
    cond_pos=0
    cond_neg=0
    pred_pos=0
    pred_neg=0
    
    for i in range(len(T)):
        
        if(np.round(T[i])==np.round(Y[i]) and np.round(T[i])==1.0):
            c_tp=c_tp+1
            
        if(np.round(T[i])==np.round(Y[i]) and np.round(T[i])==0.0):
            c_tn=c_tn+1 
            
        if(np.round(T[i])!=np.round(Y[i]) and np.round(T[i])==1.0):
            c_fn=c_fn+1
            
        if(np.round(T[i])!=np.round(Y[i]) and np.round(T[i])==0.0):
            c_fp=c_fp+1  

        if(np.round(T[i])==1.0):
            cond_pos=cond_pos+1
            
        if(np.round(T[i])==0.0):
            cond_neg=cond_neg+1   
            
        if(np.round(Y[i])==1.0):
            pred_pos=pred_pos+1
            
        if(np.round(Y[i])==0.0):
            pred_neg=pred_neg+1             
            
    TPR=c_tp/(cond_pos)
    TNR=c_tn/(cond_neg) 
    FNR=c_fn/(cond_pos)
    FPR=c_fp/(cond_neg)
    
    recall=TPR
    precision=c_tp/(pred_pos)
    F1=2*recall*precision/(recall+precision)
    
    ROC=TPR/FPR
    ROC2=TNR/FNR

    print(c_tp,c_fp)  
    print(c_fn,c_tn) 
    print('--------------')
    print('TPR=',TPR,'FPR=',FPR)
    print('FNR=',FNR,'TNR=',TNR)
    print('--------------')
    print('ROC=TPR/FPR',ROC)
    print('ROC2=TNR/FNR',ROC2)
    print('--------------')
    print('Recall=',recall,'precision=',precision,'F1_score=',F1)
    print('\n')
    print('\n')
    


