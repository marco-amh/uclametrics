# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:32:38 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv as inv

def vif(df, constant):
    X = df.copy()
    if constant==True:
        X['constant'] = 1
    else:
        pass
    
    X_names = X.columns
    N = X.shape[0] # number of obs
    M = X.shape[1] # covariates
    X = X.to_numpy() 
    vif = np.zeros((M,1)) #store results
    
    for i in np.arange(0,M):
        Y_tmp = X[:,i:i+1]
        X_tmp = np.delete(X, i, axis=1)
        betahat = inv(X_tmp.T @ X_tmp) @ (X_tmp.T @ Y_tmp) # coefficients
        ehat = Y_tmp - X_tmp @ betahat # residuals
        SSE = ehat.T @ ehat # Sum of squares of errors
        if X_names[i] == 'constant':
            SST = Y_tmp.T @  Y_tmp    # Sum of Squares Total
        else:
            SST = ( Y_tmp - np.mean(Y_tmp)).T @  (Y_tmp - np.mean(Y_tmp))   # Sum of Squares Total
        r2 = 1 - (SSE/SST)
        vif[i] = 1 / (1-r2)
    df1 = pd.DataFrame(X_names.T)
    df1.rename( columns={0 :'Variables'}, inplace=True )
    df1['VIF'] = vif
    print("VIF=1 Predictors No Correlated")
    print("1<VIF<5 Predictors Moderately Correlated")
    print("VIF>5-10 Predictors Correlated")
    
    return df1

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name='ols', index_col=0)
vif(df=df, constant=True)
'''
