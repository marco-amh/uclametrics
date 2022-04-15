# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:50:56 2022

@author: marco
"""
import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from scipy.stats import f
import math
import random
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')

# Coefficients
Y = df.iloc[:,0:1]
X = df.iloc[:,1:]
X_names = X.columns
Y = Y.to_numpy()
X = X.to_numpy()
N = X.shape[0]
M = X.shape[1]
instability = np.empty((N,M))* np.nan
erec = np.zeros((N,1))
erecols = np.zeros((N,1))
predrec = np.zeros((N,1))

for i in range(0,N):
    YY = Y[:i+1]
    XX = X[:i+1]

    betahat = inv(XX.T @ XX) @ (XX.T @ YY)
    instability[i:i+1] = betahat.ravel()
    # Recurisve residuals
    predrec[i+1] = X[i+1:i+2] @ betahat
    erecols[i+1] = Y[i+1:i+2] - X[i+1:i+2] @ betahat 
    erec[i+1]   = (Y[i+1:i+2] - X[i+1:i+2] @ betahat) / (1 + (X[i+1:i+2] @ inv(XX.T @ XX) @ X[i+1:i+2].T))**(1/2)
    

np.cumsum(erec)
def alpha(alpha):
    if alpha == 0.90:
        a = 0.850
    elif alpha == 0.95:
        a = 0.948
    elif alpha == 0.99:
        a = 1.143
    else:
        raise ValueError("alpha can only be 0.9, 0.95 or 0.99")
    return a

a = alpha(alpha=.9)
nrr = N-M
rcusumci = (a*nrr**(1/2)+2*a*np.arange(0, N-M) / nrr**(1/2)) * np.array([[-1.], [+1.]])


erec = erec[M:]
pd.DataFrame(np.cumsum(erec)).plot()

# Parameter stability test

cusum = np.zeros((erec.shape[0],5))

for i in range(0,N-M):
    cusum[i:i+1,0:1] = (1/(N-M)) * np.sum((erec[:i+1]-np.mean(erec[:i+1]))**2)
    cusum[i:i+1,1:2] =  1/cusum[i:i+1,0:1]**(1/2) * np.sum(erec[:i+1])
    cusum[i:i+1,2:3] = np.abs(cusum[i:i+1,1:2])/ (1+(2*(i-M)/(N-M)))
    cusum[i:i+1,3:4] = (0.5*0.95) * (1+(2*(i-M)/(N-M)))
    cusum[i:i+1,4:5] = -(0.5*0.95) * (1+(2*(i-M)/(N-M)))

np.max(cusum[:,2:3][cusum[:,2:3]<np.inf])

cusum = pd.DataFrame(cusum)

cusum.rename(columns={0:'recursive_var', 1:'cusum_statistic',2:'cusum_critical', 3:'bound_pos', 4:'bound_neg'}, inplace=True)

cusum[['cusum_statistic','bound_pos','bound_neg']].plot()




import statsmodels.stats.diagnostic as rec
from statsmodels.formula.api import ols
model1 = ols(formula = 'y ~ b0 +b1+b2-1', data = df)
result1 = model1.fit()
result1.summary()
recur = rec.recursive_olsresiduals(result1, skip=None, lamda=0.0, alpha=0.95, order_by=None)

pd.DataFrame(recur[5]).plot()
