# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:09:25 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.linalg import inv as inv

def chow(df, constant):
    df1 = df.copy()
    if constant==True:
        df1['cons']=1
    else:
        pass
    
    Y = df1.iloc[:,0:1].to_numpy()
    X = df1.iloc[:,1:].to_numpy()
    N = X.shape[0] # number of obs
    M = X.shape[1] # number of covariates
    
    betahat = inv(X.T @ X) @ (X.T @ Y) #  coeficient estimation
    ehat = Y - X @ betahat # residuals
    SSE_tot = ehat.T @ ehat # sum of squares error
    Chow = pd.DataFrame(index=df1.index[2:-2]) # remove the first 2 and the last 2 obs
    Chow['Chow'] = 0 # create column to store chow statistic
    j=0
    i=2
    for i in range(2,N-1):
        Y_a = Y[:i] # 1st regression's dependent variable
        X_a = X[:i] # 1st regression's covariates
        Y_b = Y[i:] # 2nd regression's dependent variable
        X_b = X[i:] # 2nd regression's covariates
        
        N_a = X_a.shape[0] # 1st regression obs
        N_b = X_b.shape[0] # 2nd regression obs
        
        betahat_a = inv(X_a.T @ X_a) @ (X_a.T @ Y_a) # 1st reg coefs
        ehat_a = Y_a - X_a @ betahat_a # 1st residuals
        SSE_a = ehat_a.T @ ehat_a # 1st fitted sum of squares of errors
        
        betahat_b = inv(X_b.T @ X_b) @ (X_b.T @ Y_b) # 2nd reg coefs
        ehat_b = (Y_b - X_b @ betahat_b) # 2nd residuals
        SSE_b = ehat_b.T @ ehat_b # 2nd fitted sum of squares of errors
        
        Chow.iloc[j:j+1,:] = ((SSE_tot - SSE_a - SSE_b)/M)/((SSE_a+SSE_b)/(N_a+N_b-2*M))
        j = j+1
        
    Chow['F_critical'] = f.ppf(0.95, M, N_a+N_b-2*M)
    Chow['p_value'] = 1- f.cdf(Chow['Chow'], M, N_a+N_b-2*M)
    
    print('Chow Test \n'
          'Run the full regression, and split that equation in two groups \n'
          'eq 1: yt = b0a + b1a +...vat \n'
          'eq 2: yt = b0b + b1b +...vbt \n'
          'Ho: b0a =b0b, b1a=b1b...=bma=bmb , Coefficients are the same for \n'
          'both regressions.\n'
          'Ha: Coefficients are different .\n' )
    
    return Chow
    

# References
#https://en.wikipedia.org/wiki/Chow_test
#https://www.r-bloggers.com/2021/11/how-to-do-chow-test-in-r/
#https://www.iuj.ac.jp/faculty/kucc625/method/panel/chow_test.pdf

'''
import os
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')
df1 = chow(df=df, constant=True)    
'''

