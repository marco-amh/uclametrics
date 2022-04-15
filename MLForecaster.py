# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:38:52 2021

@author: marco
"""
import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
os.chdir('C://Users//marco//Desktop//Projects//MLForecaster')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


model_ml = RandomForestRegressor(max_depth=12, random_state=0)
model_ml = Ridge(alpha=0.1)
model_ml = LinearRegression()
model_ml = Lasso(alpha=0.1)
model_ml = AdaBoostRegressor()
model_ml = SVR(kernel='linear', C=100, gamma=0.1)
model_ml = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
model_ml = DecisionTreeRegressor()

df = pd.read_csv('data.csv', index_col='Date')


# Create data frame to store forecastings
h = 24 #forecast horizon
date_format = '%Y-%m-%d'
dtObj = datetime.strptime(df.index[-1], date_format) # Get last observation date
future_date = (dtObj + relativedelta(months=h)).strftime(date_format) # Get future date
# Get date of the forecasts
fdates = [] # store dates of the forecasts
for i in range(1,h+1):
    fdates.append((dtObj + relativedelta(months=i)).strftime(date_format))
fdates = pd.DataFrame(fdates) # to data frame

# Create empty numpy to store results
a = np.zeros(shape=(h,df.shape[1])) # Create a m*n matrix of zeros
df1 = pd.DataFrame(a, index=fdates.loc[:].squeeze(), columns=df.columns) # to data frame
df = pd.concat([df,df1]) # concatenate obs and fcasted values

# Covariates
df.iloc[615:616,1:2]= 2136456.494

# Window rolling forecasts
s = 350 # rolling window size
w = len(df)-s+1-h+1 # number of estimations and forecasts
fcast = df.iloc[:,0:1].copy() # Matrix to store results of the autoregressive process
tmp  = df.copy() # Help matrix to keep the covariates
for i in range(1,w): # Create the columns for the forecasts
    fcast['fcast_'+ str(i)] = np.nan

# Numpy format
y = df.iloc[:,0:1].to_numpy().reshape(-1,1)
X = df.iloc[:,1:].to_numpy()
x = df.iloc[:,1:].to_numpy()

# Count set to 1
j = 1
for i in range(s,len(df)-h+1):
    mod = model_ml.fit(X[i-s:i],y[i-s:i])
    print(i)
    for k in range(0,h):
        fcast.iloc[i+k:i+k+1,fcast.columns.get_loc('fcast_'+str(j))] = mod.predict(x[i+k:i+k+1])[0]
        tmp.iloc[i+k:i+k+1,0:1] = mod.predict(x[i+k:i+k+1])[0]
        tmp.iloc[:,1:2] = tmp.iloc[:,0:1].shift(1).values  
        x = tmp.iloc[:,1:].to_numpy()
        
    x = df.iloc[:,1:].to_numpy()
    j = j+1

# Check the range inclusive function in pandas

# Create the columns for the forecasts
rmse = np.zeros(shape=(h,w-1))
rmse = pd.DataFrame(rmse, index=range(0,h), columns=fcast.columns[1:])

val_mtx = fcast.iloc[:-h,:]


list_a =[]

for i in range(1,w):
    list_a.append(((val_mtx.m - val_mtx['fcast_'+ str(i)])**2).dropna().values)

diff_mtx = pd.DataFrame(list_a)

rmse = pd.DataFrame(diff_mtx.sum(axis=0)/(diff_mtx.count(axis=0).values))**(1/2)
rmse.plot()

