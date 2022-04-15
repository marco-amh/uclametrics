# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:09:25 2022

@author: marco
"""

import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.linalg import inv as inv
import math
import os

os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

def chow(df):
    Y = df.iloc[:,0:1]
    X = df.iloc[:,1:]
    Y = Y.to_numpy()
    X = X.to_numpy()
    N = X.shape[0]
    M = X.shape[1]
    
    betahat = inv(X.T @ X) @ (X.T @ Y)
    fitted = (X @ betahat)
    ehat = (Y - X @ betahat)
    SSE_t = ehat.T @ ehat
    Chow = pd.DataFrame(index=df.index[math.ceil(N*.1):math.ceil(N*.9)])
    Chow['Chow'] = 0
    j=0
    for i in range(math.ceil(N*.1),math.ceil(N*.9)):
        Y_a = Y[:i]
        X_a = X[:i]
        Y_b = Y[i:]
        
        X_b = X[i:]
        N_a = X_a.shape[0]
        N_b = X_b.shape[0]
        betahat_a = inv(X_a.T @ X_a) @ (X_a.T @ Y_a)
        fitted_a = (X_a @ betahat_a)
        ehat_a = (Y_a - X_a @ betahat_a)
        SSE_a = ehat_a.T @ ehat_a
        
        betahat_b = inv(X_b.T @ X_b) @ (X_b.T @ Y_b)
        fitted_b = (X_b @ betahat_b)
        ehat_b = (Y_b - X_b @ betahat_b)
        SSE_b = ehat_b.T @ ehat_b
        Chow.iloc[j:j+1,:] = ((SSE_t - (SSE_a + SSE_b))/M)/((SSE_a+SSE_b)/(N_a+N_b-(2*M)))
        j = j+1
        
    Chow['F_critical'] = f.ppf(0.95, M, N_a+N_b-2*M)
    Chow['p_value'] = 1- f.cdf(Chow['Chow'], M, N_a+N_b-2*M)
    
    return Chow
    
    print('Chow Test \n'
          'Run the full regression, and split that equation in two groups \n'
          'eq 1: yt = b0a + b1a +...vat \n'
          'eq 2: yt = b0b + b1b +...vbt \n'
          'Ho: b0a =b0b, b1a=b1b...=bma=bmb , Coefficients are the same for \n'
          'both regressions.\n'
          'Ha: Coefficients are different .\n' )
    

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')
df1= chow(df=df)    
