# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:23:22 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.linalg import inv as inv

def lagger_e(p,df):
    df  = df.copy()
    df1 = pd.DataFrame() # Support matrix
    for j in range(0,df.shape[1]):
        for i in range(0,p+1):
            tmp = df.iloc[:,j:j+1].shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
    df1.columns = df1.columns.str.replace("_0", "")
    df1.drop(columns=df.columns[1:], inplace=True)
    return df1.dropna()


def anova(df):
    df['cons'] = 1
    Y = df.iloc[:,0:1].to_numpy()
    X = df.iloc[:,1:].to_numpy()
    #N = X.shape[0]
    M = X.shape[1]
    betahat = inv(X.T @ X) @ (X.T @ Y)
    #fitted = (X @ betahat)
    ehat = (Y - X @ betahat)
    SSE = ehat.T @ ehat # sum of squares of residuals
    
    return SSE, M
    
def Granger_Causality(df, lags):  
    results = pd.DataFrame(np.ones((lags,6))*np.nan)
    results.rename(columns={0:'X->Y F-calc',1:'X->Y F-critical',2:'X->Y p-value',
                            3:'Y->X F-calc',4:'Y->X F-critical',5:'Y->X p-value'}, inplace=True)
    df1 = df.copy() # x as function of y
    df2 = df1[df1.columns[::-1]].copy() # y as function of x
    
    j = 0
    for i in range(1,lags+1):
        df1_r = lagger_e(p=i,df = df1.iloc[:,0:1]) # Eq1 restricted
        df1_u = lagger_e(p=i,df = df1)             # Eq1 unrestricted (all vars)
        n = df1_u.shape[0]                         # No. observations
        
        ssr1_r, ssr1_r_m = anova(df1_r)               # Anova Eq1 restricted
        ssr1_u, ssr1_u_m = anova(df1_u)               # Anova Eq1 unrestricted
        q1 = ssr1_u_m - ssr1_r_m                      # Number of restrictions
        f1 = ((ssr1_r-ssr1_u)/q1)/(ssr1_u/(n-ssr1_u_m))# F-Test

        df2_r = lagger_e(p=i,df = df2.iloc[:,0:1])   # Eq2 restricted
        df2_u = lagger_e(p=i,df = df2)               # Eq2 unrestricted (all vars)
    
        ssr2_r, ssr2_r_m = anova(df2_r)              # Anova Eq2 restricted
        ssr2_u, ssr2_u_m = anova(df2_u)              # Anova Eq2 unrestricted
        q2 = ssr2_u_m - ssr2_r_m                     # Number of restrictions
        
        f2 = ((ssr2_r-ssr2_u)/q2)/(ssr2_u/(n-ssr2_u_m))# F-Test eq2
        F_critical = f.ppf(0.95, ssr1_u_m-1, n-ssr1_u_m)
        
        p_value_f1 = 1- f.cdf(f1, q1,n-ssr1_u_m)
        p_value_f2 = 1- f.cdf(f2, q2,n-ssr2_u_m)
        
        results.iloc[j:j+1,0:1] =  f1
        results.iloc[j:j+1,1:2] =  F_critical
        results.iloc[j:j+1,2:3] =  p_value_f1
        results.iloc[j:j+1,3:4] =  f2
        results.iloc[j:j+1,4:5] =  F_critical
        results.iloc[j:j+1,5:6] =  p_value_f2
        
        j=j+1
    print('Granger Causality Test \n'
          'Granger statistic is ((SSE_r-SSE_u)/q) / ((SSE_u)/(n-k))  \n'
          'Ho: b_1=b_2=...=0. X does not Granger cause Y.\n'
          'Ha: X Granger causes Y.\n'
          'We reject Ho if p-value < 0.05 ')
    
    return results
    
'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'ols', index_col = 0)
df = df.iloc[:,0:2]
a = Granger_Causality(df=df, lags=3)
aa = Granger_Causality(df=df, lags=4)
'''