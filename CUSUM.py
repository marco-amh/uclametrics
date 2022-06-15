# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:50:56 2022

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv as inv
import seaborn as sns
import matplotlib.pyplot as plt

def alpha(alpha):
    if alpha == 0.90:
        a = 0.850
    elif alpha == 0.95:
        a = 0.948
    elif alpha == 0.99:
        a = 1.143
    else:
        raise ValueError("alpha can only be 0.9, 0.95 or 0.99")
    return a

def Cusum(df, a, plot):
    df1 = df.copy()            # Original Data Frame copy
    Y = df1.iloc[:,0:1]        # Get the dependent variable
    X = df1.iloc[:,1:]         # Get the covariates
    X_names = X.columns        # Covariate names
    Y = Y.to_numpy()           # To numpy
    X = X.to_numpy()           # To numpy
    N = X.shape[0]             # No. of observations
    M = X.shape[1]             # No. of covariates
    parameter = np.empty((N,M))  * np.nan # Empty nan matrix of N*M
    predrec = np.zeros((N,1))    * np.nan  # Zeros vector of N*1 for one step ahead projection
    fcast_error = np.zeros((N,1))* np.nan     # Zeros vector of N*1  forecast error
    recursive = np.zeros((N,1))  * np.nan  # Zeros vector of N*1 standarized fcast error
    

    for i in range(M-1,N):
        YY = Y[:i+1]           # Increasing the window sample each iteration by one obs
        XX = X[:i+1]           # Increasing the window sample each iteration by one obs

        betahat = inv(XX.T @ XX) @ (XX.T @ YY)  # Get the OLS coefficients
        parameter[i:i+1] = betahat.ravel()      # Store the time varying coefficients
        if i <= (N-2):

        # Recursive residuals                 
            predrec[i+1] = X[i+1:i+2] @ betahat                   # One step ahead forecast
            fcast_error[i+1] = Y[i+1:i+2] - X[i+1:i+2] @ betahat  # One step ahead forecast error 
            recursive[i+1]   = (Y[i+1:i+2] - X[i+1:i+2] @ betahat) / (1 + (X[i+1:i+2] @ inv(XX.T @ XX) @ X[i+1:i+2].T))**(1/2) # One step ahead Standarized forecast error
        else:
            pass
    
    parameter = pd.DataFrame(parameter)    # Parameters 
    parameter.columns = X_names            # Rename
    
    df2 = pd.DataFrame(np.hstack([predrec, fcast_error, recursive]))
    df2.rename(columns={0:'Prediction',1:'Forecast_error',2:'Recursive_residuals'}, inplace=True)
    
    # Standarized Recursive Residuals
    df2['Standarized'] = (df2['Recursive_residuals']  * 1 / np.var(df2['Recursive_residuals'], ddof=1)**(1/2)).cumsum()
    # Note: We use N-M dof to get the cusum statistic
    
    # Test statistic
    df2['Test_statistic'] = np.abs(df2['Standarized'])/ (1+(2*(np.arange(0, N)-M)/(N-M)))    
    
    # Theoretical boundary of a Brownian motion with probability        
    a = alpha(a)       # Probability for theoretical boundaries

    df2[['Low','Up']] = np.nan
  
    df2.loc[M:,'Low'] = (a * (N-M)**(1/2) + 2 * a * np.arange(0, N-M) / (N-M)**(1/2)) * np.array([-1])
    df2.loc[M:,'Up'] =  (a * (N-M)**(1/2) + 2 * a * np.arange(0, N-M) / (N-M)**(1/2)) * np.array([1])
    
    # Copy date index as original data frame
    df2.index = df1.index[:]
        
    if plot==True:
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        line1, = plt.plot(df2.index,df2['Standarized'].values, lw=2, linestyle='-', color='purple', label='Standarized Recursive Residuals - Parameter Stability')
        line2, = plt.plot(df2.index,df2['Up'].values, lw=2, linestyle='--', color='black', label='95% confidence band')
        line3, = plt.plot(df2.index,df2['Low'].values, lw=2, linestyle='--', color='black')
        sns.despine()
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        plt.gca().set(title='CUSUM Parameter Stability', xlabel = 'Date', ylabel = 'CUSUM')
        plt.xticks(np.arange(0, N, step=round(N*.05)), rotation=90)
        plt.show() #plot       

    print('Ho: \beta_0 = \beta_1 = ... =\beta-m ')
    
    return parameter,df2

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'ols', index_col = 0)
a,b = Cusum(df=df, a=0.95, plot=True)
'''
# References
#https://www.stata.com/manuals/tsestatsbcusum.pdf
#https://www.statsmodels.org/dev/_modules/statsmodels/stats/diagnostic.html#recursive_olsresiduals
