# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:16:58 2022
@author: marco antonio martinez huerta
marco.martinez@ucla.edu
"""
import pandas as pd
import numpy as np
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from scipy.stats import f
from scipy.stats import t
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt

def Unusual(df, constant, CooksMethod,plot):
    original_index=df.index
    Y = df.iloc[:,0:1]                       # dependent variable is the first column
    X = df.iloc[:,1:]                        # independent variables
    if constant==True:
        X['ones'] = 1                        # include constant
    
    X_names = X.columns                      # covariate names
    N = X.shape[0]                           # number of observations
    M = X.shape[1]                           # number of covariates
    Y = Y.to_numpy()                         # to array format
    X = X.to_numpy()                         # to array format
    student_res = np.zeros((N,1))            # Empty array to store distance
    leverage = np.zeros((N,1))               # Empty array to store distance
    Cooks = np.zeros((N,1))                  # Empty array to store distance
    betahat = inv(X.T @ X) @ X.T @ Y         # coefficients
    fitted = X @ betahat                     # fitted values
    e  = Y - fitted                          # residuals
    mse = (e.T @ e) / (N-M)                  # Mean squared error of the model
    ps2 = (M) * mse                          # Mean squared error of the model
    for i in np.arange(0,N):                 # loop to delet one observation at time
        leverage[i] = ( X[i:i+1] @ inv(X.T @ X) @ X[i:i+1,:].T)
        #Hat=np.diag(X@inv(X.T@X)@X.T).reshape(-1,1) #This is equivalent to the previous line
        student_res[i] = e[i] / ((mse*(1-leverage[i]))**(1/2))    
        
        if CooksMethod==1:
            Y_tmp = np.delete(Y, i, axis=0)                        # delete Y's i'th obs
            X_tmp = np.delete(X, i, axis=0)                        # delete X's i'th obs
            betahat_tmp = inv(X_tmp.T @ X_tmp) @ (X_tmp.T @ Y_tmp) # coefficient re calaculated
            fitted_tmp = (X_tmp @ betahat_tmp)                     # Fitted without the ith observation
            fitted_aux = np.delete(fitted, i, axis=0)
            fitted_dif = fitted_aux - fitted_tmp                   # fitted values difference
            Cooks[i] = (fitted_dif.T @ fitted_dif) / ps2          # Distance
        
        if CooksMethod==2:
            Cooks[i] = ((e[i]**2)/ps2) * ( leverage[i] / (1-leverage[i])**2)

    
    
    df1 = pd.DataFrame(student_res)
    df1.rename( columns={0 :'StudentizedResiduals'}, inplace=True )
    df1['Threshold'] = t.ppf(0.95,N-M)
    df1['Outliers'] = np.where( df1['StudentizedResiduals'] > df1['Threshold'],1,0)
    df1.index = original_index
    
    df2 = pd.DataFrame(leverage)
    df2.rename( columns={0 :'HatMatrix'}, inplace=True )
    df2['Threshold'] = 2*M/N
    df2['Leverage'] = np.where( df2['HatMatrix'] > df2['Threshold'],1,0)
    df2.index = original_index
    
    df3 = pd.DataFrame(Cooks)
    df3.rename( columns={0 :'CooksDistance'}, inplace=True )
    df3['Threshold'] = f.ppf(0.5,M,N-M)
    df3['Influential'] = np.where( df3['CooksDistance'] > df3['Threshold'],1,0)
    df3.index = original_index
    
    if plot==True:
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        plt.scatter(df1.index, df1.iloc[:,0:1].values, lw=2, linestyle='-', color='red') 
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        sns.despine()
        plt.gca().set(title='Outliers', xlabel = '', ylabel = '')
        plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
        plt.show() #plot  
        
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        plt.scatter(df2.index, df2.iloc[:,0:1].values, lw=2, linestyle='-', color='black') 
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        sns.despine()
        plt.gca().set(title='Leverage', xlabel = '', ylabel = '')
        plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.05)), rotation=90)
        plt.show() #plot 
        
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        plt.scatter(df3.index, df3.iloc[:,0:1].values, lw=2, linestyle='-', color='purple') 
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        sns.despine()
        plt.gca().set(title='Influential', xlabel = '', ylabel = '')
        plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.05)), rotation=90)
        plt.show() #plot 

    return df1,df2,df3

def Mahalanobis(df, cov=None, pvalue=0.001, plot=False):
    df1 = df.copy()
    x = df1.to_numpy()
    M = df1.shape[1]
    
    x_mu = x - np.mean(x,axis=0)
    if not cov:
        cov = np.cov(x.T)
       
    mahal = x_mu @ inv(cov) @ x_mu.T
    
    df2 = pd.DataFrame(mahal.diagonal())
    df2.rename( columns={0 :'Mahalanobis'}, inplace=True )
    df2['pvalue'] = 1 - chi2.cdf(df2['Mahalanobis'], M-1)
    df2['Leverage'] = np.where(df2['pvalue'] <=  pvalue,1,0)
    df2.index = df1.index
    
    print('Mahalanobis distance is defined as the distance between two given points provided that they are in multivariate space. This distance is used to determine statistical analysis that contains a bunch of variables.')
    
    if plot == True:
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        plt.scatter(df2.index, df2.iloc[:,0:1].values, lw=2, linestyle='-', color='red') 
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        sns.despine()
        plt.gca().set(title='Mahalanobis Distance- Leverage', xlabel = '', ylabel = '')
        plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.1)), rotation=90)
        plt.show() #plot  
    
    return df2


# References:
    
#https://www.statisticshowto.com/cooks-distance/
#https://www.statology.org/cooks-distance-python/
#https://en.wikipedia.org/wiki/Cook%27s_distance
#https://lymielynn.medium.com/a-little-closer-to-cooks-distance-e8cc923a3250#:~:text=Simply%20said%2C%20Cook's%20D%20is,the%20removal%20of%20the%20point.
#https://stackoverflow.com/questions/46304514/access-standardized-residuals-cooks-values-hatvalues-leverage-etc-easily-i
#https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
#https://en.wikipedia.org/wiki/Mahalanobis_distance#Relationship_to_leverage
#https://www.statology.org/standardized-residuals/
#https://online.stat.psu.edu/stat462/node/172/ # for outliers
#https://online.stat.psu.edu/stat501/lesson/11/11.3#:~:text=The%20good%20thing%20about%20internally,is%20generally%20deemed%20an%20outlier.
#https://stats.stackexchange.com/questions/204708/is-studentized-residuals-v-s-standardized-residuals-in-lm-model
#https://en.wikipedia.org/wiki/Studentized_residual#:~:text=In%20statistics%2C%20a%20studentized%20residual,in%20the%20detection%20of%20outliers.
#https://en.wikipedia.org/wiki/Leverage_(statistics)
#https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3#:~:text=Mahalanobis%20Distance%20(MD)%20is%20an,center%20(see%20Formula%201).
#https://www.journaldev.com/58952/mahalanobis-distance-in-python
#
