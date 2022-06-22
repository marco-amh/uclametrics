# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:00:23 2022

@author: Marco Antonio Martinez Huerta
marco.martinez@ucla.edu
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


t = 500
R=  0.01
Q = 0.001
F = 1
mu = 0
e1 = (np.random.normal(0,1,500) * np.sqrt(R)).reshape(-1,1)
e2 = (np.random.normal(0,1,500) * np.sqrt(Q)).reshape(-1,1)
beta = np.zeros((t,1))
y = np.zeros((t,1))
x = np.random.normal(0,1,500).reshape(-1,1)
for i in range(1,t):
    beta[i:i+1,:] = beta[i-1:i,:] + e2[i:i+1,:]
    y[i:i+1,:] = beta[i:i+1,:] + e1[i:i+1,:]
    
df = np.concatenate([y,x],axis=1)
    
def Kalman_Filter(df,v,p,a=1,plot=False):
       df1 = pd.DataFrame(df.copy())
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
       mu = 0                                           # Mean
       a = a                                            # Prior coefficient of the state equation  (a < 1)
       x0 = x11 = 0                                     # Initial x_0=0. This is for part A in this question
       p11 = p00 = 1                                    # Initial variance state model
       v = v                                            # variance state space model- this term adds noise - transitory
       w = p                                            # variance state representation - this is little noise - persistent

       for i in range(0,n):
           x10 = mu + (a * x11)                         # 1st step: Estimation "state x_t" equation. Its equal to y_hat
           p10 = (a**2 * p11 ) + w                      # 2nd step: variance of the state variable

           fcast.append((a**2) * x10)                   # a^2 * \hat x_t
         
           fcast_error.append( x[i] - ((a**2) * x10) )  # forecast error: x_t - (a^2 * \hat x_t)
           feta = p10 + v                               # add variance of the presistent noise, v
           k = p10 * (1/feta)                           # Kalman Gain in time t: (a^2 * \sigma^2 + \sigma^2_w )/(a^2 * \sigma^2 + \sigma^2_w +\sigma^2_v)
           kgain.append(p10 * (1/feta))                 # Kalman Gain - cumulative
           x11 = ((1-k) * (x10*a)) + (k * x[i])         # Filtered state variable - Latent state equation of x
           p11 = p10 - k * p10                          # Variance of state variable at time t
           p_tt.append(p11)                             # Variance of state variable - appended
           x_tt.append(x11)                             # Save filters state variable
           filter_error.append(x[i] - x11)              # Residual / One-step ahead forecast error

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
           plt.gca().set(title='Kalman Filter', xlabel = 'Date', ylabel = names)
           plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
           plt.show() #plot           

       return df1
   
df1 = Kalman_Filter(df=y,p=0.090,v=.0015,a=1,plot=True)

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'M0', index_col = 0)
df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
df1 = Kalman_Filter(df=df,j=0.15,p=0.090,v=.0015,plot=True)
'''