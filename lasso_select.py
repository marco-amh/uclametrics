# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:39:44 2022

@author: marco
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import t
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from scipy.stats import norm

os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

def lasso_selection(df, method, lamb=0.01):
    names = df.columns
    Y = df.iloc[:, 0:1].to_numpy().reshape(-1,1)
    X = df.iloc[:,1:].to_numpy()
    if method =='cv':
        lasso = LassoCV(n_alphas=100, fit_intercept=False)
    elif method =='BCCH':
        # Belloni-Chen-Chernozhukov-Hansen rule
        (n,p) = X.shape
        c = 1.1
        a = 0.05
        Xscale_a = np.max(np.mean((X ** 2 * Y ** 2),axis=0)) ** 0.5
        lamb_pilot = 2*c*norm.ppf(1-a/(2*p))*Xscale_a/np.sqrt(n)
        lasso = Lasso(alpha = lamb_pilot/2, fit_intercept = False, max_iter = 10000)
        lasso.fit(X,Y)
        ehat =  (Y-np.matmul(X,lasso.coef_).reshape(-1,1))
        Xscale_b = np.max(np.mean((X ** 2 * ehat ** 2),axis=0)) ** 0.5
        lamb = 2*c*norm.ppf(1-a/(2*p))*Xscale_b/np.sqrt(n)
        print("Lambda Belloni-Chen-Chernozhukov-Hanses rule is " + str(lamb/2))
        lasso = Lasso(alpha=lamb, copy_X=True, fit_intercept=True, max_iter=1000,
                      normalize=False)
    elif method =='lambda':
         lasso = Lasso(alpha=lamb, copy_X=True, fit_intercept=True, max_iter=1000,
                      normalize=False)
    lasso.fit(X,Y.ravel())
    coefs = lasso.coef_
    numerate = np.arange(0,coefs.shape[0])
    signif = pd.DataFrame(np.where(coefs != 0, numerate,-1))
    signif = signif[signif >= 0].dropna()
    signif= signif + 1
    signif = np.append(0, signif)
    df_a = df.iloc[:, list(signif)]
    X_names = df_a.iloc[:,1:].columns
    #df_b = df_a.to_numpy()
    return df_a, coefs


#dtafile = 'Data.xlsx'
#df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')
#a, b = lasso_selection(df=df, method='lambda', lamb=0.01)
#a, b = lasso_selection(df=df, method='cv')
