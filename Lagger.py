# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:51:22 2022

@author: Marco Antonio Martinez Huerta
marco.martinez@ucla.edu
"""
import pandas as pd

def Lagger(df, p,kind):
    df1 = pd.DataFrame() # Support matrix
    if kind=='a': # Only lags, ordered by lags
        for i in range(0,p+1):
            tmp = df.shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
        df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    elif kind=='b': # Lags and contemporaneous; ordered by variable
        for j in range(0,df.shape[1]):
            for i in range(0,p+1):
                tmp = df.iloc[:,j:j+1].shift(i)
                tmp.columns = tmp.columns + '_'+ str(i)
                df1 = pd.concat([df1,tmp], axis=1)
        df1.columns = df1.columns.str.replace("_0", "")
    elif kind=='c': # Lags and contemporaneous, ordered by lags
        for i in range(0,p+1):
            tmp = df.shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
        df1.columns = df1.columns.str.replace("_0", "")
    elif kind=='d': # Only Lags; ordered by variable
        for j in range(0,df.shape[1]):
            for i in range(0,p+1):
                tmp = df.iloc[:,j:j+1].shift(i)
                tmp.columns = tmp.columns + '_'+ str(i)
                df1 = pd.concat([df1,tmp], axis=1)
        df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    elif kind=='e': # dependent contemporaneous and lags, and only lags of the covariates
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

