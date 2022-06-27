# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:31:38 2022

@author: marco
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import inv

def Kalman_Filter(df,v,p,a=1,plot=False):
       df1 = pd.DataFrame(df.copy())
       names = df1.columns[0]
       y = df.iloc[:,0:1].to_numpy()
       x = df1.iloc[:,1:].to_numpy()
       M = x.shape[1]
       N = x.shape[0]
       # Initial parameters
       fcast = np.ones((N,1)) * np.nan
       fcast_error = np.ones((N,1)) * np.nan
       kgain = np.ones((N,1)) * np.nan                   
       p_tt = np.ones((N,M,M)) * np.nan
       beta_tt= np.ones((N,M)) * np.nan
       filter_error = np.ones((N,1)) * np.nan
       n = len(x)
       mu = 0                                           # Mean
       beta0 = beta11 = np.zeros((1,M))                 # Initial x_0=0. This is for part A in this question
       
       a = np.eye(M,M)                                  # Prior coefficient of the state equation  (a < 1)  
       w11 = w00 =np.eye(M,M)                           # Initial variance state model
       v = v                                            # variance state space model- this term adds noise - transitory
       w = p                                            # variance state representation - this is little noise - persistent
       

       
        
       for i in range(0,n):
           beta10 = mu + ( beta11 @ a.T)                         # 1st step: Estimation "state x_t" equation. Its equal to y_hat
           
           yhat = (x[i:i+1,:] @ beta10.T).T       # changed - beta10                # a^2 * \hat x_t
           residual = y[i:i+1,:] - yhat                               # forecast error
           
           w10 = (a @ w11 @ a.T ) + w                      # 2nd step: variance of the state variable
           v10 = (x[i:i+1,:] @ w10.T @ x[i:i+1,:].T) + v                               # add variance of the presistent noise, v
           
           k = (w10 @ x[i:i+1,:].T) @ inv(v10)                          # Kalman Gain in time t: (a^2 * \sigma^2 + \sigma^2_w )/(a^2 * \sigma^2 + \sigma^2_w +\sigma^2_v)
           beta11 = (beta10.T + (k * residual.T)).T        # Filtered state variable - Latent state equation of x
           w11 = w10 - (k @ x[i:i+1,:] @ w10)                          # Variance of state variable at time t
  
 
           fcast[i:i+1,:] = yhat                    
           fcast_error[i:i+1,:]     = residual 
           #kgain.append(w10 @ inv(v10)   )                # Kalman Gain - cumulative
           p_tt[i:i+1,:,:]          = w11                            # Variance of state variable - appended
           beta_tt[i:i+1,:]         = beta11                             # Save filters state variable
           filter_error[i:i+1,:]    = residual              # Residual / One-step ahead forecast error

        # End of Kalman Filter       
       beta2 = np.zeros((N,M))                          # hold the draw of the state variable
       wa = np.random.normal(0,1,size=(N, M))                     
       i = N                                            # period N
       p00 = p_tt[-1]
       beta2[i-1:i,:] = beta_tt[i-1:i,:] + (wa[i-1:i,:] @ np.linalg.cholesky(p00))


       for i in reversed(range(0,N-1)):
           pt = p_tt[i] 
           bm = beta_tt[i:i+1,:] + (pt @ a.T @ inv(a @ pt @ a.T + p) @ (beta2[i+1:i+1+1,:] - mu - beta_tt[i:i+1,:] @ a.T).T).T
           pm = pt - pt @ a.T @ inv( a @ pt @ a.T + p) @ a @ pt
           beta2[i:i+1,:] = bm + (wa[i:i+1,:] @ np.linalg.cholesky(pm))



       #x = x
       fcast = pd.DataFrame(fcast)
       
       fcast_error =pd.DataFrame(fcast_error)
       #kgain =pd.DataFrame(kgain)
       #p_tt = pd.DataFrame(p_tt)
       beta_tt = pd.DataFrame(beta_tt, index=df.index)
       filter_error = pd.DataFrame(filter_error)
      ## df1[str(df1.columns[0])+'_gap'] = x-beta_tt
       #df1[str(df1.columns[0])+'_filtered'] = x_tt
       #df1[str(df1.columns[0])+'_gap'] = x-x_tt
      ## df1[str(df1.columns[0])+'_variance_state'] = p_tt
      ## df1[str(df1.columns[0])+'_gain'] = kgain
       #df1[str(df1.columns[0])+'_fcast_h1'] = fcast
       #print('the kalman gain is: ', str(kgain[-1]))


       
       if plot == True:
           plt.figure(figsize=(16,8))
           sns.set_style('ticks')
           line, = plt.plot(df1.index,beta, lw=2, linestyle='-', color='black', label='True beta')
           line, = plt.plot(df1.index,beta_tt, lw=2, linestyle='-', color='red', label='Estimated State')
           
           plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
           plt.axhline(y=0, color='black', linestyle='-')
           sns.despine()
           plt.gca().set(title='Kalman Filter', xlabel = 'Date', ylabel = names)
           plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
           plt.show() #plot           

       return beta_tt, beta2, fcast


#np.random.seed(430)
t = 500
v =  0.01
w1 = 0.001
w2 = 0.05
w3 = 0.00005
e1 = (np.random.normal(0,1,t) * np.sqrt(v)).reshape(-1,1)
e2 = (np.random.normal(0,1,t) * np.sqrt(w1)).reshape(-1,1)
e3 = (np.random.normal(0,1,t) * np.sqrt(w2)).reshape(-1,1)
e4 = (np.random.normal(0,1,t) * np.sqrt(w3)).reshape(-1,1)

beta1 = np.zeros((t,1))
beta2 = np.zeros((t,1))
beta3 = np.zeros((t,1))

y = np.zeros((t,1))
x1 = np.random.normal(0,1,t).reshape(-1,1)
x2 = np.random.normal(0,1,t).reshape(-1,1)
x3 = np.random.normal(0,1,t).reshape(-1,1)

for i in range(1,t):
    beta1[i:i+1,:] = beta1[i-1:i,:] + e2[i:i+1,:]
    beta2[i:i+1,:] = beta2[i-1:i,:] + e3[i:i+1,:]
    beta3[i:i+1,:] = beta3[i-1:i,:] + e4[i:i+1,:]
    y[i:i+1,:] = (x1[i:i+1,:] * beta1[i:i+1,:]) + (x2[i:i+1,:] * beta2[i:i+1,:] ) +(x3[i:i+1,:] * beta3[i:i+1,:] )+ e1[i:i+1,:] 
    
    
df = pd.DataFrame(np.concatenate([y,x1,x2,x3],axis=1))

p =  np.diag([w1,w2,w3])

df1,df2,df3 = Kalman_Filter(df=df, v=v, p = p, a=1, plot=False)
#df1.plot()
b1 = pd.concat([pd.DataFrame(beta1),df1.iloc[:,0:1],pd.DataFrame(df2[:,0:1])],axis=1)
b2 = pd.concat([pd.DataFrame(beta2),df1.iloc[:,1:2],pd.DataFrame(df2[:,1:2])],axis=1)
b3 = pd.concat([pd.DataFrame(beta3),df1.iloc[:,2:3],pd.DataFrame(df2[:,2:3])],axis=1)
b4 = pd.concat([pd.DataFrame(y),df3],axis=1)

# Beta 1 state space
plt.figure(figsize=(16,8))
sns.set_style('ticks')
line, = plt.plot(df1.index,b1.iloc[:,0:1].values, lw=2, linestyle='-', color='black', label='True beta')
line, = plt.plot(df1.index,b1.iloc[:,1:2].values, lw=2, linestyle='-', color='red', label='Estimated State')
line, = plt.plot(df1.index,b1.iloc[:,2:3].values, lw=2, linestyle='-', color='purple', label='Carter and Kohn')
plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
plt.axhline(y=0, color='black', linestyle='-')
sns.despine()
plt.gca().set(title='Kalman Filter', xlabel = 'Date')
plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
plt.show() #plot           

# Beta 2 state space
plt.figure(figsize=(16,8))
sns.set_style('ticks')
line, = plt.plot(df1.index,b2.iloc[:,0:1].values, lw=2, linestyle='-', color='black', label='True beta')
line, = plt.plot(df1.index,b2.iloc[:,1:2].values, lw=2, linestyle='-', color='red', label='Estimated State')
line, = plt.plot(df1.index,b2.iloc[:,2:3].values, lw=2, linestyle='-', color='purple', label='Carter and Kohn')
plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
plt.axhline(y=0, color='black', linestyle='-')
sns.despine()
plt.gca().set(title='Kalman Filter', xlabel = 'Date')
plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
plt.show() #plot         

# Beta 3 state space
plt.figure(figsize=(16,8))
sns.set_style('ticks')
line, = plt.plot(df1.index,b3.iloc[:,0:1].values, lw=2, linestyle='-', color='black', label='True beta')
line, = plt.plot(df1.index,b3.iloc[:,1:2].values, lw=2, linestyle='-', color='red', label='Estimated State')
line, = plt.plot(df1.index,b3.iloc[:,2:3].values, lw=2, linestyle='-', color='purple', label='Carter and Kohn')
plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
plt.axhline(y=0, color='black', linestyle='-')
sns.despine()
plt.gca().set(title='Kalman Filter', xlabel = 'Date')
plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
plt.show() #plot   

# Filter state space
plt.figure(figsize=(16,8))
sns.set_style('ticks')
line, = plt.plot(df1.index,b4.iloc[:,0:1].values, lw=2, linestyle='-', color='black', label='Observed')
line, = plt.plot(df1.index,b4.iloc[:,1:2].values, lw=2, linestyle='-', color='red', label='Estimated State')
plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
plt.axhline(y=0, color='black', linestyle='-')
sns.despine()
plt.gca().set(title='Kalman Filter', xlabel = 'Date')
plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
plt.show() #plot 
