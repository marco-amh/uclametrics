# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:03:20 2022

@author: marco
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import inv as inv
from scipy.linalg import pinv as pinv
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
os.chdir('C:/Users/marco/Desktop/Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

def decompose(df, freq ='monthly', method='multiplicative'):
    df.rename(columns={df.columns[0]:'yt'}, inplace=True)
    df1 = df.copy()

    if freq== 'monthly':
        df1['tt'] = df1.rolling(window = 12,center=True).mean()
        df1['tt'] = df1.tt.rolling(window = 2,center=True).mean()
        df1['tt'] = df1['tt'].shift(-1)
        
        # Linear least squares trend for missing values at end points
        df1['seq'] = np.arange(0,len(df1))
        df1['cons'] = 1

        dft = df1.dropna()
        Y = dft.tt.to_numpy().reshape(-1,1)
        X = dft[['cons','seq']].to_numpy()
        Ya = Y[:12]
        Xa = X[:12]
        beta1 = inv(Xa.T @ Xa) @ (Xa.T @ Ya)
        Yb = Y[len(Y)-12:]
        Xb = X[len(X)-12:]
        beta2 = inv(Xb.T @ Xb) @ (Xb.T @ Yb)
        df1['hat_low'] = beta1[0]+(df1['seq'].values*beta1[1])
        df1['hat_high'] = beta2[0]+(df1['seq'].values*beta2[1])
        df1['tt'][0:6]=df1['hat_low'][0:6]
        df1['tt'][len(df1)-6:]=df1['hat_high'][len(df1)-6:]
        df1.drop(columns=(['seq','cons','hat_low','hat_high']), inplace=True)
        
        # Detrended series
        if method == 'additive':
            df1['detrend'] = df1['yt'] - df1['tt'] 
        elif method == 'multiplicative':
            df1['detrend'] = df1['yt'] / df1['tt'] 
        df2 = df1['detrend'].copy()
        # Seasonal component
        seas_mean = pd.DataFrame(df2.groupby(df2.index.month).mean())
        seas_mean.reset_index(inplace=True)
        seas_mean.rename(columns={seas_mean.columns[0]:'month'}, inplace=True)
        seas_mean.rename(columns={seas_mean.columns[1]:'st'}, inplace=True)
        df1['month'] = pd.DatetimeIndex(df1.index).month
        df_tmp = pd.merge(df1,seas_mean, on=['month'], how='left')
        df1['st'] = df_tmp['st'].values
        df1.drop(columns=(['month','detrend']), inplace=True)

        if method == 'additive':
            # Remainder
            df1['rt'] = df1['yt'] - df1['tt'] - df1['st']
            # Seasonal adjusted
            df1['sa'] = df1['yt'] - df1['st']
        elif method == 'multiplicative':
            df1['rt'] = df1['yt'] / (df1['tt'] * df1['st'] )
            print(df1['rt'])
            # Seasonal adjusted
            df1['sa'] = df1['yt'] / df1['st']
                  
            
    if freq== 'quarterly':
        df1['tt'] = df1.rolling(window = 4,center=True).mean()
        df1['tt'] = df1.tt.rolling(window = 2,center=True).mean()
        df1['tt'] = df1['tt'].shift(-1)
        
        # Linear least squares trend for missing values at end points
        df1['seq'] = np.arange(0,len(df1))
        df1['cons'] = 1
        dft = df1.dropna()
        Y = dft.tt.to_numpy().reshape(-1,1)
        X = dft[['cons','seq']].to_numpy()
        Ya = Y[:4]
        Xa = X[:4]
        beta1 = inv(Xa.T @ Xa) @ (Xa.T @ Ya)
        Yb = Y[len(Y)-4:]
        Xb = X[len(X)-4:]
        beta2 = inv(Xb.T @ Xb) @ (Xb.T @ Yb)
        df1['hat_low'] = beta1[0]+(df1['seq'].values*beta1[1])
        df1['hat_high'] = beta2[0]+(df1['seq'].values*beta2[1])
        df1['tt'][0:3] = df1['hat_low'][0:3]
        df1['tt'][len(df1)-3:] = df1['hat_high'][len(df1)-3:]
        df1.drop(columns = (['seq','cons','hat_low','hat_high']), inplace=True)
        
        # Detrended series
        df1['detrend'] = df1['yt'] - df1['tt'] 
        df2 = df1['detrend'].copy()
        
        # Seasonal component
        seas_mean = pd.DataFrame(df2.groupby(df2.index.month).mean())
        seas_mean.reset_index(inplace=True)
        seas_mean.rename(columns={seas_mean.columns[0]:'month'}, inplace=True)
        seas_mean.rename(columns={seas_mean.columns[1]:'st'}, inplace=True)
        df1['month'] = pd.DatetimeIndex(df1.index).month
        df_tmp = pd.merge(df1,seas_mean, on=['month'], how='left')
        df1['st'] = df_tmp['st'].values
        df1.drop(columns=(['month','detrend']), inplace=True)
      
        if method == 'additive':
            # Remainder
            df1['rt'] = df1['yt'] - df1['tt'] - df1['st']
            # Seasonal adjusted
            df1['sa'] = df1['yt'] - df1['st']
        elif method == 'multiplicative':
            df1['rt'] = df1['yt'] / (df1['tt'] * df1['st'] )
            print(df1['rt'])
            # Seasonal adjusted
            df1['sa'] = df1['yt'] / df1['st']
        
    return df1

'''
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='holt-winters')
df1 = decompose(df=df, freq ='monthly', method='multiplicative')

df1['rt'].plot()

df2 = seasonal_decompose(df, model='multiplicative', extrapolate_trend='freq')
df2.resid.plot()

# The endpoint is not the same as the stats decompose function because
# at they take out of the regression the last observation, whereas we
# use the last j observations for the linear trend.
#3.781944892576754, 1750.7559463869343
#25.035919225630348, 1600.8608903890306
'''
