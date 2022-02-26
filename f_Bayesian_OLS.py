# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:27:00 2022

@author: marco
"""
import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as inv
import matplotlib.pyplot as plt
os.chdir('C://Users//marco//Desktop//Projects//Bayesian_OLS')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

#y = Puller.Banxico(serie="SR16734", name="IGAE", plot=False)
#p = Puller.Banxico(serie="SP1", name="Inflation", plot=False)
#r = Puller.Banxico(serie="SF3270", name="Interest_rate", plot=False)
#m = Puller.Banxico(serie="SF1", name="Money", plot=False)
#df = pd.concat([y, p, r, m], axis=1).dropna()
df = pd.read_excel('inflation.xls', index_col=0)

Y=df.iloc[:,0:1].to_numpy()
X=df.iloc[:,1:].to_numpy()

def plot_dist(df, column, title):
    mu = df[column].mean()
    sigma = df[column].std()
    n, bins, patches = plt.hist(x=df[column], bins='auto', 
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel(None)
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if (maxfreq % 10 > 0) else maxfreq + 10)
    plt.show() #plot
    
def ar_companion_matrix(beta):
    # dont include constant
    k = beta.shape[0]-1
    FF = np.zeros((k, k))
    #insert identity matrix
    FF[1:k, 0:(k-1)] = np.eye(N=k-1, M=k-1)
    temp = (beta[1:k+1, :]).T
    #state space companion form
    #Insert coeffcients along top row
    FF[0:1,0:k+1] = temp
    return(FF)

def gibbs(X,Y,reps,burn,t0,d0,plot):
    reps = reps # number of Gibbs iterations
    burn = burn # percent of burn-in iterations
    out  = np.zeros((reps, X.shape[1]+1))
    t1  = Y.shape[0]  #number of observations
    b0 =  np.zeros((X.shape[1],1))#Priors
    sigma0 = np.eye((X.shape[1])) # variance matrix
    # priors for sigma2
    t0 = t0
    d0 = d0
    # Starting values
    B = b0
    sigma2 = 1
    for i in range(0,reps):
        M = inv(inv(sigma0) + (1/sigma2) * X.T @ X) @ (inv(sigma0) @ b0 + (1/sigma2)* X.T @ Y)
        V = inv(inv(sigma0) + (1/sigma2) * X.T @ X)
        chck = -1
        while (chck < 1):
            B = M + (np.random.normal(0,1,X.shape[1]) @ np.linalg.cholesky(V)).T.reshape(-1,1)
            b = ar_companion_matrix(B)
            ee = np.max(np.abs(np.linalg.eig(b)[1]))
            if (ee <= 1):
                chck = 1
        # compute residuals
        resids = Y - X @ B
        T2 = t0 + t1
        D1 = d0 + resids.T @ resids
        # keep samples after burn period
        out[i,] = np.append(B.T,sigma2)
        #draw from Inverse Gamma
        z0 = np.random.normal(1,1,t1)
        z0z0 = z0.T @ z0
        sigma2 = D1/z0z0
    
    out = pd.DataFrame(out[burn:reps,:])
    
    if plot==True:
        for i in range(0,out.shape[1]):
            if i != out.shape[1]-1:
                plot_dist(df=out, column=[i], title='Estimator distribution of beta ' + str(i))
            else:
                plot_dist(df=out, column=[i], title='Estimator distribution of the variance')
        
    return(out)


gibbs(X=X,Y=Y,reps=5000,burn=4000,t0 = 1,d0 = 0.1, plot=True)



'''
gibbs_sampler <- function(X,Y,B0,sigma0,sigma2,theta0,D0,reps,out,out1){
  for (i in 1:reps){
    
    M = solve(solve(sigma0)
              +as.numeric(1/sigma2)*t(X)%*%X)%*%(solve(sigma0)%*%b0
              +as.numeric(1/sigma2)*t(X)%*%Y)
    V = solve(solve(sigma0)+as.numeric(1/sigma2)*t(X)%*%X)
    chck=-1
    while (chck < 1){ #check for stability
      
      B= M+t(rnorm(ncol(X))%*%chol(V))
      b = ar_companion_matrix(B)
      ee <- max(sapply(eigen(b)$values,abs))
      if(ee<=1){
      }
    }
    # compute residuals
    resids <- Y- X%*%B
    T2 = t0 + t1
    D1 = d0 + t(resids) %*% resids
    
    # keeps samples after burn period
    out[i,] <- t(matrix(c(t(B),sigma2)))
    
    #draw from Inverse Gamma
    z0 = rnorm(t1,1)
    z0z0 = t(z0) %*% z0
    sigma2 = D1/z0z0
    
    # compute 2 year forecasts
    yhat = rep(0,h)
    end = as.numeric(length(Y))
    #yhat[1:2] = Y[(end-1):end,]
    cfactor = sqrt(sigma2)
    X_mat = c(1,rep(0,ncol(X)-1))
    
    
    for(m in ncol(X):h){
      for (lag in 1:(ncol(X)-1)){
        #create X matrix with p lags
        X_mat[(lag+1)] = yhat[m-lag]
      }
      # Use X matrix to forecast yhat
      yhat[m] = X_mat %*% B + rnorm(1) * cfactor
    }
    
    out1[i,] <- yhat
  }
  return = list(out,out1)
}
    
    
# Set the parameters
reps = 5000 # number of Gibbs iterations
burn = 4000 # percent of burn-in iterations

# Forecast horizon
h = 24

# Matrix to store posterior coefficients and forecasts
out  = matrix(0, nrow = reps, ncol = ncol(X_BO) + 1)
out1 = matrix(0, nrow = reps, ncol = h)

t1 <- nrow(Y_CA) #number of observations
b0 = matrix(0,ncol(X_CA),1) #Priors
sigma0 <- diag(ncol(X_CA)) # variance matrix
# priors for sigma2
t0= 1
d0=0.1

# Starting values
B = b0
sigma2 = 1
df1 = gibbs_sampler(X_CA,Y_CA, B,sigma0,sigma2,t0,d0,reps,out,out1)

coef <- results[[1]][(burn+1):reps,]
forecasts <- results[[2]][(burn+1):reps,]


const <- mean(coef[,1])
beta1 <- mean(coef[,2])
beta2 <- mean(coef[,3])
beta3 <- mean(coef[,4])
sigma <- mean(coef[,5])