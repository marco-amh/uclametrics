# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:32:38 2022

@author: marco
"""
import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


def vif(df):
    X = df.iloc[:,1:]
    X_names = X.columns
    N = X.shape[0]
    M = X.shape[1]
    X = X.to_numpy()
    vif = np.zeros((M,1))
    
    for i in np.arange(0,M):
        Y_tmp = X[:,i:i+1]
        X_tmp = np.delete(X, i, axis=1)
        betahat = inv(X_tmp.T @ X_tmp) @ (X_tmp.T @ Y_tmp)
        # Fitted
        fitted = (X_tmp @ betahat)
        # Residuals
        SSE = np.sum((Y_tmp - X_tmp @ betahat)**2)
        # Sum of Squares Total
        SST = np.sum((Y_tmp - np.mean(Y_tmp) )**2)
        r2 = 1 - (SSE/SST)
        vif[i] =1/(1-r2)
    vif = pd.DataFrame(vif)
    vif.rename( columns={0 :'VIF'}, inplace=True )
    print("VIF=1 Predictors No Correlated")
    print("1<VIF<5 Predictors Moderately Correlated")
    print("VIF>5-10 Predictors Correlated")
    
    return vif

'''
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')
vif(df=df)
'''
