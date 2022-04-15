# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:59:45 2022

@author: marco
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
import warnings                                  # `do not disturbe` mode
from dateutil.relativedelta import relativedelta
from datetime import datetime
warnings.filterwarnings('ignore')
os.chdir('C:/Users/marco/Desktop/Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

def hw(df,a,b,g,seas):
    if seas == 'monthly':
        s = 11
        sp = 1
        sb = 12
    elif seas == 'quarterly':
        s = 3
        sp = 3
        sb= 4
    df1 = df.copy()
    month_mean = df1.groupby(df1.index.month).mean().to_numpy()
    n = df1.shape[0]
    yt = df1.copy().to_numpy()
    df1.index = pd.to_datetime(df1.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    dtObj = datetime.strptime(df1.index[0], '%Y-%m-%d') # Get last observation date
    #last_date = (dtObj - relativedelta(months=sp)).strftime('%Y-%m-%d')
    last_date = (dtObj).strftime('%Y-%m-%d')
    last_date = pd.to_datetime(last_date, format = '%Y-%m-%d')

    
    lt = np.zeros((n+1,1))
    bt = np.zeros((n+1,1))
    st = np.zeros((n+sb,1))
    hw = np.zeros((n+1,1))
    lt[0] = month_mean[last_date.month-sp]
    
    bt[0] = np.exp( ((np.log(np.sum(yt[sb:sb * 2,:])/sb))-(np.log(np.sum(yt[:sb,:]) /sb))) /sb)
    st[0:sb] =  yt[0:sb]/lt[0]
    hw[0] = np.nan

    for i in range(0,n):
        lt[i+1] = a * (yt[i] - st[i]) + (1-a) * (lt[i]+bt[i])
        bt[i+1] = b * (lt[i+1] - lt[i]) + (1-b) * bt[i]
        st[i+sb] = g * (yt[i] - lt[i] - bt[i]) + (1-g) * st[i]
        hw[i+1] = lt[i+1] + (1 * bt[i+1]) + st[i+1]

    df1['Holt-Winters']= hw[:n]
    df1.plot(rot=90)
    df1['L']= lt[1:n+1]
    df1['B']= bt[1:n+1]
    df1['S']= st[sb:n+sb+1]

    return df1

#dtafile = 'Data.xlsx'
#dfa = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='Disaggregation')
#dfb = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='holt-winters')

#df3= hw(df=dfa,a=.71,b=.66,g=0.23, seas='quarterly')
#df4= hw(df=dfb,a=.71,b=.66,g=0.23, seas='monthly')

# https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572