# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:00:17 2022

@author: marco
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.linalg import inv as inv
from scipy.linalg import pinv as pinv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
os.chdir('C:/Users/marco/Desktop/Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='holt-winters')

def stl(df):
    df.rename(columns={df.columns[0]:'yt'}, inplace=True)
    df1 = df.copy()

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

    df1['sa'] = df1['yt'] - df1['st']
    
    # Local Regression (LOESS)
    yt = df1['yt'].to_numpy().reshape(-1,1)
    sa = df1['sa'].to_numpy().reshape(-1,1)
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(yt.reshape(-1,1))
    y_scaled = scaler.transform(sa.reshape(-1,1))

    X_poly = PolynomialFeatures(5)
    X_poly = X_poly.fit_transform(sa)
    # Linear regression
    model = LinearRegression()
    model = model.fit(X_poly, y_scaled)

    # Loess trend
    trend = model.predict(X_poly)
    df1['trend'] = scaler.inverse_transform(trend)


    # Remainder
    df1['rt'] = df1['yt'] - df1['trend'] - df1['st']

    return df1

df1 = stl(df=df)

df1[['yt','trend']].plot()
df1[['yt','tt']].plot()
df1[['trend','tt']].plot()
df1['trend'].plot()
df1['st'].plot()
df1['rt'].plot()
    
