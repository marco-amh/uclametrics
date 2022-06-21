# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:22:07 2022

@author: marco
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
import scipy.signal as ss
import matplotlib.patches as mpatches
import matplotlib.font_manager as font_manager

def cross_corr(df,lags,k,B, plot):
    random.seed(a=430)
    df1 = df.copy()
    names = df1.columns
    n = df1.shape[0]
    k = k #size of blocks
    B = B #number of bootstraps
    s = math.ceil(n/k) #number of blocks in each bootstraps
    ccf_bs = np.zeros((B,(lags*2)+1))# Matrix to store the results
    #X = df1.iloc[:,0:1].to_numpy()
    #Y = df1.iloc[:,1:2].to_numpy()
    df_tmp = df1.to_numpy()
    y=df_tmp[:,0:1]
    x=df_tmp[:,1:2]
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    length = (len(result) - 1) // 2
    lo = length - lags
    hi = length + (lags + 1)
    cc = result[lo:hi]
    for i in range(0,B):
        tmp = np.zeros((s*k, 2))
        for j in range(1,s+1):
            tn = random.sample(range(k,n+1), 1)[0] #last point of time
            #print(tn)
            tmp[(j-1)*k:j*k , :] =  df_tmp[tn-k:tn,:] #fill the boots vector with observations in a block  
       # Function
        Y = tmp[:, 0:1]
        X = tmp[:,1:]
        result = ss.correlate(Y - np.mean(Y), X - np.mean(X), method='direct') / (np.std(Y) * np.std(X) * len(Y))
        length = (len(result) - 1) // 2
        lo = length - lags
        hi = length + (lags + 1)
        ccf_coefs = result[lo:hi]
        ccf_bs[i:i+1, :] = ccf_coefs.ravel()
        
    
    coefs_bs = pd.DataFrame(ccf_bs)
    
    g_std = np.std(ccf_bs, axis=0).reshape(-1,1)
    
    print('La variable dinámica es '+ str(names[0]))
     
    if plot==True:
        banxicofont = {'fontname':'Calibri'}
        font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
        plt.figure(figsize=(11.8/2.54, 10.27/2.54))
        g_x = np.arange(-lags,lags+1)
        #g_lo = np.quantile(ccf_bs, .05, axis=0).ravel()# one s.d. low 
        #g_me = np.quantile(ccf_bs, .50, axis=0).ravel() # median
        #g_hi = np.quantile(ccf_bs, .95, axis=0).ravel()# one s.d. high
        g_lo = (cc + (1.96*g_std)).ravel()# one s.d. low 
        g_hi = (cc - (1.96*g_std)).ravel()# one s.d. high
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.fill_between(g_x, g_lo,g_hi, facecolor='#CBC3E3', interpolate=False)
        ax.plot(g_x, cc, color='red', lw=2, label='Mean')
        #ax.plot(g_x, g_me, color='#52307c', lw=2, label='Median')
        ax.axvline(x = 0, color = 'black', linestyle = '--')
        ax.axhline(y = 0, color = 'black', linestyle = '-')
        ax.set_xticks(g_x)
        patch = mpatches.Patch(color='red', label=str(names[0])+'(t +/- j)')
        plt.legend(handles=[patch],edgecolor = 'white', title = '', loc = 3)
        
        plt.ylim([-1,  1])
        sns.despine()
        #ax.set_xticklabels('df['Time']')
        plt.show()
        
    return cc, coefs_bs
  
'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name='Taylor', index_col=0)
df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
a,b = cross_corr(df=df.iloc[:,0:2],lags=7,k=2,B=100, plot=True) 
'''