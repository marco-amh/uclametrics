# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:15:07 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv as inv
import matplotlib.pyplot as plt
import seaborn as sns

def Decomposition(df, freq ='monthly', method='multiplicative', plot=False):
    df1 = df.copy()
    N = df1.shape[0]
    name = df1.columns[0]
    df1.index = pd.to_datetime(pd.to_datetime(df1.index, format = '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S'))
    
    if freq== 'monthly':
        w1 = 12 # rolling window 1
        w2 = 2      # rolling window 2
        months = 12
        w3 = 6 # first few observations
    elif freq== 'quarterly':
        w1 = 4 # rolling window 1
        w2 = 2      # rolling window 2
        months = 4
        w3 = 3 # first few observations
    
    df1['Trend'] = df1.rolling(window = w1,center=True).mean()
    df1['Trend'] = df1.Trend.rolling(window = w2,center=True).mean()
    df1['Trend'] = df1['Trend'].shift(-1)
        
    # Linear least squares trend for missing values at end points
    df1['seq'] = np.arange(0,N)
    df1['cons'] = 1

    dft = df1.dropna()
    Y = dft.Trend.to_numpy().reshape(-1,1)
    X = dft[['cons','seq']].to_numpy()
    Ya = Y[:months]
    Xa = X[:months]
    beta1 = inv(Xa.T @ Xa) @ (Xa.T @ Ya)
    Yb = Y[len(Y)-months:]
    Xb = X[len(X)-months:]
    beta2 = inv(Xb.T @ Xb) @ (Xb.T @ Yb)                    # coefficients
    df1['hat_low']  = beta1[0]+(df1['seq'].values*beta1[1])
    df1['hat_high'] = beta2[0]+(df1['seq'].values*beta2[1])
    df1['Trend'][:w3]=df1['hat_low'][:w3]
    df1['Trend'][len(df1)-w3:]=df1['hat_high'][N-w3:]
    df1.drop(columns=(['seq','cons','hat_low','hat_high']), inplace=True)
        
    # Detrended series
    if method == 'additive':
        df1['detrend'] = df1[name].values - df1['Trend'].values
    elif method == 'multiplicative':
        df1['detrend'] = (df1[name].values /  df1['Trend'].values)
    
    df2 = df1['detrend'].copy()
    
    # Seasonal component
    seas_mean = pd.DataFrame(df2.groupby(df2.index.month).mean())
    seas_mean.reset_index(inplace=True)
    seas_mean.rename(columns={seas_mean.columns[0]:'month'}, inplace=True)
    seas_mean.rename(columns={seas_mean.columns[1]:'Seasonal'}, inplace=True)
    df1['month'] = pd.DatetimeIndex(df1.index).month
    df_tmp = pd.merge(df1,seas_mean, on=['month'], how='left')
    df1['Seasonal'] = df_tmp['Seasonal'].values
    df1.drop(columns=(['month','detrend']), inplace=True)

    if method == 'additive':
        # Remainder
        df1['Remainder'] = df1[name] - df1['Trend'] - df1['Seasonal']
        # Seasonal adjusted
        df1['Seasonal_adjusted'] = df1[name] - df1['Seasonal']
    elif method == 'multiplicative':
        # Remainder
        df1['Remainder'] = df1[name] / (df1['Trend'] * df1['Seasonal'] )
        # Seasonal adjusted
        df1['Seasonal_adjusted'] = df1[name] / df1['Seasonal']
    
    names = df1.columns
    df1.index = pd.to_datetime(df1.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    
    trend_strength    = 1- (np.var(df1['Remainder']) / np.var(df1['Remainder']+df1['Trend']))
    seasonal_strength = 1- (np.var(df1['Remainder']) / np.var(df1['Remainder']+df1['Seasonal']))
    
    print('The strength of the trend component is ' +  str(round(trend_strength,2) ))
    print('The strength of the seasonal component is ' +  str(round(seasonal_strength,2) ))
    
    if plot== True:
        for i in range(0,5):
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            if i+1 ==5:
                plt.plot(df1.index,df1.iloc[:,0:1].values, lw=2, linestyle='-', label=names[0])
                plt.plot(df1.index,df1.iloc[:,4:5].values, lw=2, linestyle='-', label=names[4])
            else:
                plt.plot(df1.index,df1.iloc[:,i:i+1].values, lw=2, linestyle='-', label=names[i])
            sns.despine()
            plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
            plt.gca().set(title='Classical Decomposition - ' , xlabel = 'Date')
            plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
            plt.show() #plot
    
    return df1

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
dfa = pd.read_excel(url, sheet_name = 'M0', index_col = 0)
dfb = pd.read_excel(url, sheet_name = 'boe1', index_col = 0)
a = Decomposition(df=dfa, freq ='monthly', method='additive', plot=True)
b = Decomposition(df=dfa, freq ='monthly', method='multiplicative', plot=True)
c = Decomposition(df=dfb.iloc[:,0:1], freq ='quarterly', method='additive', plot=True)
d = Decomposition(df=dfb.iloc[:,0:1], freq ='quarterly', method='multiplicative', plot=True)
'''

#Notes
# The endpoint is not the same as the stats decompose function because
# at they take out of the regression the last observation, whereas we
# use the last j observations for the linear trend.
#3.781944892576754, 1750.7559463869343
#25.035919225630348, 1600.8608903890306