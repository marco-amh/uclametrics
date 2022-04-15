# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:23:22 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.linalg import inv as inv
import os

os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

# REVISAR LA HIPÓTESIS Y SI ES CON EL SSRegression con el SSError


dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')
df = df.iloc[:,0:2]

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
    Y = df.iloc[:,0:1]
    X = df.iloc[:,1:]
    Y = Y.to_numpy()
    X = X.to_numpy()
    N = X.shape[0]
    M = X.shape[1]
    betahat = inv(X.T @ X) @ (X.T @ Y)
    fitted = (X @ betahat)
    ehat = (Y - X @ betahat)
    SSR = (fitted-np.mean(Y)).T @ (fitted-np.mean(Y))
    
    return SSR, M
    
def granger(df, lags):
    df1 = df.copy()
    df2 = df1[df1.columns[::-1]]
    n = df1.shape[0]
    
    df1_r = lagger_e(p=lags,df = df1.iloc[:,0:1])
    df1_u = lagger_e(p=lags,df = df1)
    
    ssr1_r, ssr1_r_m = anova(df1_r)
    ssr1_u, ssr1_u_m = anova(df1_u)
    q1 = ssr1_u_m - ssr1_r_m 
    f1 = ((ssr1_r-ssr1_u)/q1)/(ssr1_u/(n-ssr1_u_m))
    
    df2_r = lagger_e(p=lags,df = df2.iloc[:,0:1])
    df2_u = lagger_e(p=lags,df = df2)
    
    ssr2_r, ssr2_r_m = anova(df2_r)
    ssr2_u, ssr2_u_m = anova(df2_u)
    q2 = ssr2_u_m - ssr2_r_m 
    f2 = ((ssr2_r-ssr2_u)/q2)/(ssr2_u/(n-ssr2_u_m))
    
    F_critical = f.ppf(0.95, ssr1_u_m-1, n-ssr1_u_m)
    p_value_f1 = 1- f.cdf(f1, ssr1_u_m-1,n-ssr1_u_m)
    p_value_f2 = 1- f.cdf(f1, ssr1_u_m-1,n-ssr1_u_m)
    
    print('Granger Causality Test \n'
          'Granger statistic is ((SSR_r-SSR_u)/q) / ((SSR_u)/(n-k))  \n'
          'Ho: b_1=b_2=...=0. X does not Granger cause Y.\n'
          'Ha: X Granger causes Y.\n'
          'We reject Ho if p-value < 0.05 \n' 
          'X Granger causes Y?'
          'F1 calculated = ' + str(round(f1[0][0],5)) + '\n' + 
          'F1 critical value = ' + str(round(F_critical,5)) +'\n'+
          'F1 p-value = ' + str(round(p_value_f1[0][0],5)) +'\n'+
          'Y Granger causes X? \n'
          'F2 calculated = ' + str(round(f2[0][0],5)) + '\n' + 
          'F2 critical value = ' + str(round(F_critical,5)) +'\n'+
          'F2 p-value = ' + str(round(p_value_f2[0][0],5)))
    
    
df3 = granger(df=df, lags=10)
