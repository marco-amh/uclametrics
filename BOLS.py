# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:27:00 2022

@author: marco antonio martinez huerta
"""

import pandas as pd
import numpy as np
from scipy.linalg import inv
from scipy.linalg import cholesky # upper traingular
import matplotlib.pyplot as plt
import warnings                             # `do not disturbe` mode
warnings.filterwarnings('ignore')


# Code base on the BoE bayesian econometrics for central bankers handbook. pgs 55-57

###################################### Lags ##################################
def lagger_a(p,df):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = df.shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    return df1.dropna()
################################### VARs #####################################
url = 'https://github.com/marco-amh/uclametrics/blob/main/Data.xlsx?raw=true'
df0 = pd.read_excel(url, sheet_name = 'BoE1', index_col = 0)
df0.index = pd.to_datetime(df0.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

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

