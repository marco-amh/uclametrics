# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:17:08 2022

@author: marco
"""

import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from sklearn.preprocessing import PolynomialFeatures
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

def hpfilter(df,lam):
    df1 = df.copy()
    df2 = df.copy()
    y = df1.to_numpy()
    n = y.shape[0]
    k1 = np.eye(n,n)
    k2 = np.eye(n, k=1) * -2
    k3 = np.eye(n, k=2)
    k = k1 + k2 + k3
    k = k[:-2,:]
    I = np.identity(n)
    w1 = lam
    t1 = inv (I + w1 * k.T @ k ) @ y
    
    lamb = np.full(shape=n,fill_value=lam,dtype=np.int)
    lamb[0]=lam*3
    lamb[-1] = lam*3
    lamb[1]=lam*(3/2)
    lamb[-2] = lam*(3/2)

    w2 = lamb.reshape(-1,1)
    t2 = inv (I + w2 * k.T @ k ) @ y

    df1['trend'] = t1
    df1['cycle'] = df1['y']-df1['trend']
    df2['trend_esc'] = t2
    df2['cycle_esc'] = df2['y']-df2['trend_esc']
    return (df1,df2)


def hpfilter_(df, lam):
    y = df.copy()
    n = df.to_numpy().shape[0]
    w=lam #lambda

    a = 1 + (6 * w) 
    b = -4 * w
    c = w
    d = np.array([c,b,a])
    d = np.ones((n,1))*d
    m = np.diag(d[:,2]) + np.diag(d[0:n-1,1],1)+ np.diag(d[0:n-1,1],-1)
    m = m + np.diag(d[0:n-2,0],2) + np.diag(d[0:n-2,0],-2)

    m[0,0] = 1 + w
    m[0,1] = -2 * w
    m[1,0] = -2 * w
    m[1,1] = 5*w+1;
    m[n-1-1,n-1-1] = (5 * w) +1
    m[n-1-1,n-1] = -2 * w
    m[n-1,n-1-1] = -2 * w
    m[n-1,n-1] = 1 + w

    s = pd.DataFrame(inv(m) @ y)
    df['trend'] =s.values
    df['cycle'] = df['y']-df['trend']
    return(df)


'''
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='Univariate')

dfa,dfb = hpfilter(df=df, lam=14000)

dfa[['y','trend']] .plot()
dfb[['y','trend_esc']] .plot()


df2 = hpfilter_(df=df, lam=12000)

df2.plot()

# Reference:
#https://www.bankofcanada.ca/wp-content/uploads/2010/01/tr79.pdf
#A Modification of the HP Filter Aiming at Reducing the End-Point Bias

'''