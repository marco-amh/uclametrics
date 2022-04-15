# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:03:02 2022

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


RandomForestRegressor(max_depth=12, random_state=0)
LinearRegression()
Lasso(alpha=0.1)
AdaBoostRegressor()
SVR(kernel='linear', C=100, gamma=0.1)
GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
DecisionTreeRegressor()
Ridge(alpha=0.1)
LinearRegression(fit_intercept=False)


y = Puller.Banxico(serie="SR16734", name="IGAE", plot=False)
p = Puller.Banxico(serie="SP1", name="Inflation", plot=False)
m = Puller.Banxico(serie="SF1", name="Money", plot=False)
i = Puller.Banxico(serie="SF40823", name="Interest_Rate", plot=False)

df = pd.concat([y, p, m,i], axis=1).dropna()


df_tmp = df.copy() # Original matrix
names = df.columns

##############################################################################
############################## Generate lags #################################
##############################################################################
p = 4
def lagger(p,mtx):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = mtx.shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    return(df1.dropna())

##############################################################################
# This section is to create the last testing sample of the last observation ##
x_test = df.copy()
arr = pd.DataFrame(np.random.rand(1,names.shape[0]))
arr.columns= names
x_test = x_test.append(arr, ignore_index = True)
x_test = lagger(mtx=x_test, p=p)
##############################################################################
df = lagger(mtx=df, p=p)
n = df.shape[0] # number of observations

##############################################################################
################# Create data frame to store forecasts #######################
##############################################################################
h = 12 # forecast horizon
date_format = '%Y-%m-%d' # date format
dtObj = datetime.strptime(df.index[-1], date_format) # Get last observation date
future_date = (dtObj + relativedelta(months=h)).strftime(date_format) # Get future dates
fdates = [] # store dates of the forecasts

for i in range(1,h+1):
    fdates.append((dtObj + relativedelta(months=i)).strftime(date_format)) # add the months

fdates = pd.DataFrame(fdates) # to data frame

# Create empty numpy to store estimations only
a1 = np.zeros(shape=(h,df.shape[1])) # Create a mxn matrix of zeros
df1= pd.DataFrame(a1, index=fdates.loc[:].squeeze(1), columns=df.columns) # get the additional rows for the forecast horizon
df = pd.concat([df,df1]) # concatenate obs and fcasted values
df2 = df.iloc[:len(df),:names.shape[0]]
df2.columns=names
df_tmp = pd.concat([df_tmp.iloc[:p,:],df2], axis=0, ignore_index=0) # concatenate obs and fcasted values

# This part is to add the exogenous covariate
df['Cons'] = 1 # vector of constants
ft = df.iloc[:len(df_tmp)-h-p,:] # for estimating the models

##############################################################################
############# Creating the Forecasts Matrices for each model #################
##############################################################################

# Window rolling forecasts
s = 300 # rolling window size
w = len(ft)-s+1+1 # number of estimations and forecasts

# Matrix to store results of the autoregressive process
for i in range(0,names.shape[0]):
    exec(f'fcast_{i} = pd.DataFrame(df.loc[:,names[{i}] +"_"+ str(0)].copy())') 
    #exec(f'fcast_{i} = pd.DataFrame(ft.loc[:,names[{i}] +"_"+ str(0)].copy())') 

# Create the w predictions columns
for j in range(0,names.shape[0]):
    for i in range(1,w): # Create the columns for the forecasts
        exec(f'fcast_{j}["fcast_{i}"] = np.nan')

# Create dependent variables vectors
for r in range(0,names.shape[0]):
    exec(f'y_{r} = ft.iloc[: ,{r}:{r}+1].to_numpy()') 
    
# Create covariate matrices in numpy. This is for the training test
X = ft.iloc[: ,len(names):].to_numpy()
# This is for the out of sample part
z = ft.iloc[: ,len(names):]
x_test['Cons']=1
z = z.append(x_test.iloc[x_test.shape[0]-1:x_test.shape[0],names.shape[0]:]).to_numpy()

############
j=1
#for i in range(s,len(df)-h+1)
#LinearRegression(fit_intercept=False)
for i in range(s,n+1):
    if (j % 10==0):
        print(j)
    for v in range(0,names.shape[0]):
            exec(f'mod_{v} = LinearRegression(fit_intercept=False).fit(X[{i}-s:{i}],y_{v}[{i}-s:{i}])')
    tmp = df_tmp.copy()
    x = z.copy()
    for k in range(0,h):
        for v in range(0,names.shape[0]):
            exec(f'fcast_{v}.iloc[i+{k}:{i}+{k}+1,fcast_{v}.columns.get_loc("fcast_" + str(j))] = mod_{v}.predict(x[{i}+{k}:{i}+{k}+1])[0]')
            exec(f'tmp.iloc[{i}+{k}+{p}:{i}+{k}+1+{p},{v}:{v}+1] = mod_{v}.predict(x[{i}+{k}:{i}+{k}+1])[0]')

        df3 = pd.DataFrame()
        for t in range(0,p+1):
            tmp_ = tmp.shift(t)
            df3 = pd.concat([df3,tmp_], axis=1)
            df3 = df3.dropna()
        df3['Cons'] = 1
        x = df3.iloc[:,names.shape[0]:].to_numpy()

    j = j+1


# Create the columns for the forecasts
for t in range(0,names.shape[0]):
    rmse = np.zeros(shape=(h,w-1))
    rmse = pd.DataFrame(rmse, index=range(1,h+1), columns=fcast_0.columns[1:])
    exec(f'val_mtx = fcast_{t}.iloc[:-h,:] ') 
    list_a=[]  # store results of the metrics
    for i in range(1,w):
        list_a.append(((val_mtx.iloc[:,0:1].squeeze(1) - val_mtx['fcast_'+ str(i)])**2).dropna().values)
        diff_mtx = pd.DataFrame(list_a)
        exec(f'rmse_{t} = pd.DataFrame(diff_mtx.sum(axis=0)/(diff_mtx.count(axis=0).values))**(1/2) ')
    
    plt.figure(figsize=(16,8))
    plt.style.use('ggplot')
    exec(f'line, = plt.plot(range(1,h+1),rmse_{t}.values, lw=2, linestyle="-", color="b")')
    plt.gca().set(title="RMSE " + str(names[t]), xlabel = "Date", ylabel = "RMSE ")
    plt.xticks(np.arange(1, h+1, step=1), rotation=0)
    plt.show() #plot