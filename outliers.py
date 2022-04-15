# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:16:58 2022

@author: marco
"""
import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from scipy.stats import f
from scipy.stats import chi2

os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

def cooks(df):
    Y = df.iloc[:,0:1]
    X = df.iloc[:,1:]
    X_names = X.columns
    N = X.shape[0]
    M = X.shape[1]
    Y = Y.to_numpy()
    X = X.to_numpy()
    D = np.zeros((N,1))
    betahat = inv(X.T @ X) @ (X.T @ Y)
    # Fitted
    fitted = (X @ betahat)

    for i in np.arange(0,N):
        
        Y_tmp = np.delete(Y, i, axis=0)
        X_tmp = np.delete(X, i, axis=0)
        betahat_tmp = inv(X_tmp.T @ X_tmp) @ (X_tmp.T @ Y_tmp)
        # Fitted without the i observation
        fitted_tmp = (X_tmp @ betahat_tmp)
        # Residuals
        ps2 = M * (np.sum((Y_tmp - X_tmp @ betahat)**2)/(N-M))
        # Original Fitted without the i observation
        fitted_original_tmp = np.delete(fitted, i, axis=0)
        D[i] = (np.sum(fitted_original_tmp - fitted_tmp)**2) / ps2

    D = pd.DataFrame(D)
    D.rename( columns={0 :'Cooks_Distance'}, inplace=True )
    out = np.where(D>f.ppf(0.5,N,N-M))[0]
    out=pd.DataFrame(out)
    D.plot.bar(rot=90, xticks=None)

#D^Mahalanobis = (x-m)^T C^-1 (x-m)

def mahalanobis(df=None, cov=None, pvalue=0.001):
    df1 = df.copy()
    x = df.to_numpy()
    M = df1.shape[1]
    
    x_mu = x - np.mean(x)
    if not cov:
        cov = np.cov(x.T)
    
    mahal = x_mu @ inv(cov) @ x_mu.T
    
    df1['mahalanobis'] = mahal.diagonal()
    df1['p'] = 1 - chi2.cdf(df1['mahalanobis'], M-1)
    outlier = np.where(df1['p']<= pvalue)[0]
    return df1, outlier


'''
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')

dfd,dfe = mahalanobis(df=df)
dfc = cooks(df=df)

'''
