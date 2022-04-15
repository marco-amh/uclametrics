# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:22:07 2022

@author: marco
"""

import pandas as pd
import numpy as np
import os
import math
import random
import statsmodels.tsa.stattools as cc
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as ss
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


dtafile = 'Data.xlsx'

df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')


def ccf(x, y, lag_max = 100):
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]



'''
df_tmp = df.copy()
df_names = df1.columns
n = df1.shape[0] #sample size
comb = list(combinations(df1.columns,2))
'''


def cross_corr(df,lags,k,B):
    random.seed(a=430)
    df1 = df.copy()
    n= df1.shape[0]
    k = k #size of blocks
    B = B #number of bootstraps
    s = math.ceil(n/k) #number of blocks in each bootstraps
    ccf_bs = np.zeros((B,(lags*2)+1))# Matrix to store the results
    #X = df1.iloc[:,0:1].to_numpy()
    #Y = df1.iloc[:,1:2].to_numpy()
    df_tmp = df1.to_numpy()
    for i in range(0,B):
        tmp = np.zeros((s*k, 2))
        for j in range(1,s+1):
            tn = random.sample(range(k,n+1), 1)[0] #last point of time
            tmp[(j-1)*k:j*k , :] =  df_tmp[tn-k:tn,:] #fill the boots vector with observations in a block  
       # Function
        Y = tmp[:, 0:1].reshape(-1,1)
        X = tmp[:,1:]
        ccf_coefs = ccf(x=X, y=Y, lag_max = lags)
        ccf_bs[i:i+1, :] = ccf_coefs.ravel()
    
    coefs_bs = pd.DataFrame(ccf_bs)
    return coefs_bs
  
df1 = cross_corr(df=df.iloc[:,0:2],lags=3,k=2,B=100)    
np.mean(df1, axis=0)

X1= df.iloc[:,0:1].to_numpy()
Y2= df.iloc[:,1:2].to_numpy()
ccf(x=X1, y=Y2, lag_max = 3)
