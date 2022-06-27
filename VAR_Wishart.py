# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:52:06 2022

@author: D14371
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv
from scipy.linalg import cholesky # upper traingular
import matplotlib.pyplot as plt
import os
import warnings                             # `do not disturbe` mode

warnings.filterwarnings('ignore')
os.chdir('T://Marco//BOE//')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

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
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df0 = pd.read_excel(url, sheet_name = 'boe2_wishart', index_col = 0)
df0.index = pd.to_datetime(df0.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

np.random.seed(430)

p = 2 # number of lags
h = 60 # horizon
x = lagger_a(p=p,df=df0) # covariates lags
y = df0.iloc[p:,:] #endogenous

# Constant
x['cons'] = 1
first_column = x.pop('cons') # pop this column
x.insert(0, 'cons', first_column) # to the first position

N = y.shape[1] # number of endogenous variables
m = x.shape[1] # number of covariate lags
T = x.shape[0] # number of observations

df = pd.concat([y,x], axis=1, ignore_index=False).dropna()
df.plot(subplots=True)

# Compute standard deviation of each series residual via an ols 
# regression to be used in setting the prior

sigmaP = np.zeros((N,1)) # standar deviation of the S.E. vector

for i in range(0,N):
    df1 = df0.iloc[:,i:i+1]
    df1['cons'] = 1
    df1['lag'] =  df0.iloc[:,i:i+1].shift(1)
    df1 = df1.iloc[p:,:]
    Y = df1.iloc[:,0:1].to_numpy()
    X = df1.iloc[:,1:].to_numpy()
    u = Y - X @ inv(X.T @ X) @ X.T @ Y   # coefficients
    s = ((u.T @ u)/(len(u)-2))**(1/2)     # std of the standard error
    sigmaP[i] = s

# Parameters to control the prior
lamda0 = 1
lamda1 = 0.1 #tightness prior on the AR coefficients
lamda3 = 0.05 # tightness of prior on higher lags
lamda4 = 1 # tightness of prior on the constant term
# Specify the prior mean of the coefficients of the two equations of the VAR


B0 = np.zeros((N*p+1,N))

for i in range(1,N+1):
    B0[i:i+1,i-1:i] = 0.95

B0 = B0.reshape(-1,1, order='F')

Sbar = np.zeros((N,N))
Hbar = np.eye(N*p+1,N*p+1)

hbar = []
hbar.append((lamda0*lamda4)**2)
for j in range (0,len(sigmaP)):
    hbar.append((((lamda0*lamda1)/sigmaP[j])**2)[0])
for r in range(1,p):
    for j in range (0,len(sigmaP)):
        hbar.append((((lamda0*lamda1)/((2**lamda3)*sigmaP[j]))**2)[0])


np.fill_diagonal(Sbar,(sigmaP/lamda0)**2)
np.fill_diagonal(Hbar,np.array(hbar))

H = np.kron(Sbar,Hbar)

# Change priors
H[2,2] = 1e-9
H[3,3] = 1e-9
H[4,4] = 1e-9
H[6,6] = 1e-9
H[7,7] = 1e-9
H[8,8] = 1e-9


# prior scale matrix for sigma the VAR covariance
S = np.eye(N)
# prior degrees of freedom
alpha = N+1
#starting values for the Gibbs sampling algorithm
Sigma = np.eye(N)

# 0. To array form
Y = df.iloc[:,:N].to_numpy()
X = df.iloc[:,N:].to_numpy()
# 1. OLS estimation
betaols = (inv(X.T @ X) @ X.T @ Y).reshape(-1,1, order='F')   # coefficients/#'F' means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest
#Y_hat = X @ pi                # fitted values
#u = Y - Y_hat                 # residuals
#Omega = ((u.T @ u)/(T-p))     # omega



Reps = 40000
burn = 30000
out1=np.zeros((h,1)) # will store IRF of R 
out2=np.zeros((h,1)) # will store IRF of GB 
out3=np.zeros((h,1)) # will store IRF of U 
out4=np.zeros((h,1))# will store IRF of P 


def stability(betas,n,l):
    coef = betas.reshape(N*p+1, N, order='F')
    ff = np.zeros((N*p, N*p))
    ff[N+1-1:(N*p), :(N*(p-1))] = np.eye(N*(p-1),N*(p-1))
    temp = betas.reshape(N * p+1, N, order='F')
    temp = temp[ 1: N*p+1, :N].T
    ff[ :N, : N*p] = temp
    ee = np.max(np.abs(np.linalg.eig(ff)[0]))
    S = ee > 1
    return S

def IWPQ(v,ixpx):
    k = ixpx.shape[0]
    z = np.zeros((v,k))
    mu = np.zeros((k,1))
    for i in range(0,v):
        z[i,:]=(cholesky(ixpx).T @ np.random.normal(0,1, size=(k,1))).T
    out= inv(z.T @ z)
    return out

i=1
j=1

for j in range(1,Reps):
    #step 1 draw the VAR coefficients
    M = inv(inv(H) + np.kron(inv(Sigma), X.T @ X)) @ (inv(H) @ B0 + np.kron(inv(Sigma),X.T @ X) @ betaols)
    V = inv(inv(H) + np.kron(inv(Sigma), X.T @ X))
    #check for stability of the VAR
    check = -1
    while check < 0:
        beta = M + (np.random.randn(1, N*(N*p+1)) @ cholesky(V)).T
        #beta = M + (np.random.normal(0,1,size=(1, N*(N*p+1))) @ cholesky(V)).T
        ch = stability(beta,N,p)

        if ch==0:
            check = 10

    print(j)
    

    #draw sigma from the IW distribution
    e = Y - (X @ beta.reshape(N*p+1,N, order='F'))
    #scale matrix
    scale = (e.T @ e) + S
    Sigma = IWPQ(T+alpha, inv(scale))

    if j > burn:
    #impulse response using a cholesky decomposition
        A0 = cholesky(Sigma)
        v = np.zeros((h,N))
        v[p,1] = -1 # shock the government bondyield
        yhat = np.zeros((h,N))
        for i in range(p,h):
            yhat[i,:] = np.hstack([0, yhat[i-1:i,:][0],yhat[i-2:i-1,:][0]]) @ beta.reshape(N*p+1,N, order='F') + v[i,:] @ A0

        out1 = np.hstack((out1, yhat[:, 0:1].reshape(-1,1)))
        out2 = np.hstack((out2, yhat[:, 1:2].reshape(-1,1)))
        out3 = np.hstack((out3, yhat[:, 2:3].reshape(-1,1)))
        out4 = np.hstack((out4, yhat[:, 3:4].reshape(-1,1)))
    
out1 = out1[p:,1:]
out2 = out2[p:,1:]
out3 = out3[p:,1:]
out4 = out4[p:,1:]


for i in np.arange(1,N+1):
    g_x = np.arange(len(out1))
    exec(f'g_lo = np.quantile(out{i}, .16, axis=1).ravel()')# one s.d. low 
    exec(f'g_me = np.quantile(out{i}, .50, axis=1).ravel()') # median
    exec(f'g_hi = np.quantile(out{i}, .84, axis=1).ravel()')# one s.d. high
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    #colors = plt.cm.Reds(np.linspace(0.3, 0.8, 4))
    ax.fill_between(g_x, g_lo,g_hi, facecolor='#CBC3E3', interpolate=False)
    ax.plot(g_x, g_me, color='#52307c', lw=2, label='Median')
    ax.axhline(y = 0, color = 'red', linestyle = '--')
    ax.set_xticks(g_x)
    #ax.set_xticklabels('df['Time']')
    plt.show()


# Notes:
# T is traspose and is ' in matlab
# inv is the inverse or inv in matlab
# @ is matrix multiplication or * in matlab