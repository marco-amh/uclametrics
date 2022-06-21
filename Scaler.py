# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:43:02 2022

@author: marco Antonio Martínez Huerta
marco.martinez@ucla.edu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Scaler(df, method, plot):
    df1 = df.copy()
    names = df1.columns
    X = df1.to_numpy()
    muhat = np.mean(X,axis = 0).reshape(1,X.shape[1])
    stdhat = np.std(X,axis = 0).reshape(1,X.shape[1])
    maxhat = np.max(X,axis = 0).reshape(1,X.shape[1])
    minhat = np.min(X,axis = 0).reshape(1,X.shape[1])
    rangehat = maxhat-minhat
    
    if method == 'normalize_mean':
        df2 = pd.DataFrame(np.nan_to_num((X - muhat)/rangehat))
    elif method == 'normalize_min':
        df2 = pd.DataFrame(np.nan_to_num((X - minhat)/rangehat))
    elif method == 'standarize':
        df2 = pd.DataFrame(np.nan_to_num((X - muhat)/stdhat))
    
    df2.columns = names
    df2.index = df1.index
    
    if plot == True:
       plt.figure(figsize=(16,8))
       sns.set_style('ticks')
       for i in range(0,df1.shape[1]):
           plt.plot(df2.index,df2.iloc[:,i:i+1].values, lw=2, linestyle='-', label=names[i])
       plt.axhline(y=0, color='black', linestyle='-')
       sns.despine()
       plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
       plt.gca().set(title='Normalized Data', xlabel = 'Date')
       plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.05)), rotation=90)
       plt.show() #plot           
        
    return df2

def Unscaler(df_original, df_transformed, method, plot):
    # Get descriptive metrics
    X = df_original.to_numpy() 
    muhat = np.mean(X,axis = 0).reshape(1,X.shape[1])
    stdhat = np.std(X,axis = 0).reshape(1,X.shape[1])
    maxhat = np.max(X,axis = 0).reshape(1,X.shape[1])
    minhat = np.min(X,axis = 0).reshape(1,X.shape[1])
    rangehat = maxhat-minhat
    # Revert transformation
    df1 = df_transformed.copy()
    names = df1.columns
    
    if method == 'normalize_mean':
        df2 = (df1 * rangehat ) + muhat
    elif method == 'normalize_min':
        df2 = (df1 * rangehat ) + minhat
    elif method == 'standarize':
        df2 = (df1 * stdhat ) + muhat
        
    if plot == True:
       plt.figure(figsize=(16,8))
       sns.set_style('ticks')
       for i in range(0,df1.shape[1]):
           plt.plot(df2.index,df2.iloc[:,i:i+1].values, lw=2, linestyle='-', label=names[i])
       sns.despine()
       plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
       plt.gca().set(title='Normalized Data', xlabel = 'Date')
       plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.05)), rotation=90)
       plt.show() #plot           

    return df2

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'ols', index_col = 0)

df4a = Scaler(df=df, method='standarize', plot=True)
df4b = Scaler(df=df, method='normalize_mean', plot=True)
df4c = Scaler(df=df, method='normalize_min', plot=True)

df5a = Unscaler(df_original=df,df_transformed=df4a, method='standarize', plot=True)
df5b = Unscaler(df_original=df,df_transformed=df4b, method='normalize_mean', plot=True)
df5c = Unscaler(df_original=df,df_transformed=df4c, method='normalize_min', plot=True)
'''
