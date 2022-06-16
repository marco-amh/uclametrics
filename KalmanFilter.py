# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:00:23 2022

@author: marco
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def Kalman_Filter(df,j,p,v,plot):
       df1 = pd.DataFrame(df)
       names = df1.columns[0]
       x = df1[df1.columns[0]]
                
       # Initial parameters
       fcast = []
       fcast_error = []
       kgain = []
       p_tt = []
       x_tt=[]
       filter_error = []
       n = len(x)
       mu = 0 #mean
       a = j #less than 1
       x0 = x11 = 0 # initial x_0=0. This is for part A in this question
       p11 = p00 = 1 #initial variance state model
       v = v # variance state space model
       w = p # variance state representation

       for i in range(0,n):
           x10 = mu + (x11 * a) # 1st step: Estimation "state x_t" equation. Its equal to y_hat
           p10 = (a**2 * p11 ) + w # 2nd step: variance of the state variable
           fcast.append((a**2) * x10)
           fcast_error.append( x[i] - ((a**2) * x10) )# forecast error
           feta = p10 + v # add variance residual v
           k = p10 * (1/feta)
           kgain.append(p10 * (1/feta)) # Kalman Gain
           x11 = ((1-k) * (x10*a)) + (k * x[i]) # filtered state variable
           p11 = p10 - k * p10 #  variance of state variable
           p_tt.append(p11) #  variance of state variable
           x_tt.append(x11) # store filters state variable
           filter_error.append(x[i] - x11) # residual

       #x = x
       df1[str(df1.columns[0])+'_filtered'] = x_tt
       df1[str(df1.columns[0])+'_gap'] = x-x_tt
       df1[str(df1.columns[0])+'_variance_state'] = p_tt
       df1[str(df1.columns[0])+'_gain'] = kgain
       df1[str(df1.columns[0])+'_fcast_h1'] = fcast
       print('the kalman gain is: ', str(kgain[-1]))
       if plot == True:
           plt.figure(figsize=(16,8))
           sns.set_style('ticks')
           line, = plt.plot(x.index,x, lw=2, linestyle='-', color='black', label=names)
           line, = plt.plot(x.index,df1[str(df1.columns[0])+'_filtered'].values, lw=2, linestyle='-', color='red', label='State')
           line, = plt.plot(x.index,df1[str(df1.columns[0])+'_gap'].values, lw=2, linestyle='-', color='blue', label='Gap')
           plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
           plt.axhline(y=0, color='black', linestyle='-')
           sns.despine()
           plt.gca().set(title='Hodrick-Prescott Filter', xlabel = 'Date', ylabel = names)
           plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
           plt.show() #plot           

       return df1
'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'M0', index_col = 0)
df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
df1 = Kalman_Filter(df=df,j=0.15,p=0.090,v=.0015,plot=True)
'''