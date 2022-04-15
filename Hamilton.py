# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:39:48 2022

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

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='Univariate')


df2 = hamilton(h=8,p=4, df=df)

def hamilton(df,h,p):
    df  = df.copy()
    df1 = pd.DataFrame() # Support matrix
    for i in range(h,h+p):
        tmp = df.iloc[:,:].shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    df1.columns = df1.columns.str.replace("_0", "")
    df1.drop(columns=df.columns[1:], inplace=True)
    df1 = pd.concat([df,df1], axis=1)
    df1['cons']=1
    return df1
    
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