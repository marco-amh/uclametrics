# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:39:48 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv as inv
import matplotlib.pyplot as plt
import seaborn as sns

def Hamilton_Filter(df,h,p, plot):
    df2  = df.copy()
    names = df2.columns[0]
    df1 = pd.DataFrame() # Support matrix
    for i in range(h,h+p):
        tmp = df.iloc[:,:].shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    df1.columns = df1.columns.str.replace("_0", "")
    df1.drop(columns=df.columns[1:], inplace=True)
    df1 = pd.concat([df,df1], axis=1)
    df1['cons']=1
    df1.dropna(inplace=True)
    
    Y = df1.iloc[:,0:1].to_numpy()
    X = df1.iloc[:,1:].to_numpy()
    N = X.shape[0] # number of obs
    M = X.shape[1] # number of covariates
    
    df2 = df2.iloc[p+h-1:]
    
    betahat = inv(X.T @ X) @ (X.T @ Y) #  coeficient estimation
    df2['trend'] = X @ betahat # fitted values, trend
    df2['cycle'] = Y - X @ betahat # residuals, cycle
    if plot == True:
       plt.figure(figsize=(16,8))
       sns.set_style('ticks')
       line1, = plt.plot(df2.index,df2[names].values, lw=2, linestyle='-', color='black', label=names)
       line2, = plt.plot(df2.index,df2['trend'].values, lw=2, linestyle='-', color='red', label='Trend')
       line3, = plt.plot(df2.index,df2['cycle'].values, lw=2, linestyle='-', color='blue', label='Cycle')
       plt.axhline(y=0, color='black', linestyle='-')
       sns.despine()
       plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
       plt.gca().set(title='Hamilton Filter', xlabel = 'Date', ylabel = names)
       plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
       plt.show() #plot           

    return df2
    

# References
#https://quanteconpy.readthedocs.io/en/latest/_modules/quantecon/filter.html#hamilton_filter

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'M0', index_col = 0)
df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
a = Hamilton_Filter(df=df,h=24,p=25, plot=True)
'''