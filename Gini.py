# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:56:11 2022

@author: Marco Antonio Martinez Huerta
marco.martinez@ucla.edu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Gini(df,include_negatives=False,plot=False):
    df = df.to_numpy()                     # to numpy array
    df = np.sort(df,axis=0)                # sort the data in ascending [::-1] is used when descending
    N = df.shape[0]                        # number of observations
    
    if include_negatives == False:
        cum = np.cumsum(df)
        slope = cum.max() /N                     # slope (y2-y1) / (x2-x1)
        area = (N * cum.max())/2                 # (base \times height) / 2
        mtx = np.zeros((len(df),1)) * np.nan     # empty matrix to store the Slope-Lorenz
    
        for i in range(0,len(df)):
            mtx[i:i+1,:] = (slope * (i+1)) - cum[i:i+1] # difference between slope and the Lorenz curve

        lorenz = np.sum(mtx)                 # sum of the difference (area above Lorenz curve)
        gini = lorenz / area                 # Gini is the proportion fo the Lorenz area to the total area 
        
    elif include_negatives == True:
        cum, cum_a, cum_b = np.cumsum(df), np.cumsum(df), np.cumsum(df) # cumulative sum
        cum_a = np.where(cum_a < 0, 0, cum_a)  # keep the positives
        cum_b = np.where(cum_b > 0, 0, cum_b)  # keep the negatives
        slope = np.cumsum(df).max()/N          # slope of the area (y2-y1) / (x2-x1)
        area_1 = N * (np.cumsum(df).max())/2   # (base \times height) / 2
        mtx = np.zeros((len(df),1))*np.nan     # empty matrix to store the Slope-Lorenz
    
        for i in range(0,len(df)):
            mtx[i:i+1,:]= slope * (i+1) - cum_a[i:i+1] # difference between slope and the Lorenz curve

        lorenz_1 = np.sum(mtx)                 # sum of the difference (area above Lorenz curve)
        lorenz_2 = abs(np.sum(cum_b))          # area below zero and the negative Lorenz curve
        area_2 = 0 - cum.min()* len(cum)       # base times heigth- rectangle area for negative data
        gini = (lorenz_1+lorenz_2)/(area_1+area_2) # Gini is the proportion fo the Lorenz area to the total area
    
   
    print("The Gini coefficient is " + str(round(gini,4)))
    
    ## Draw a Lorenz Curve            
    df_order = df.copy()

    if plot == True and  include_negatives==True:
        cumulative = np.append(0, np.cumsum(df_order))
        plt.figure(figsize = (16,8))
        plt.plot([0,len(df)], cumulative[[0,len(df)]])
        plt.plot([0,len(df)],[0,0], '--k')
        plt.plot([0,0],[min(cumulative), 0],'--k')
        plt.plot([len(df),len(df)],[min(cumulative), max(cumulative)],'--k')
        plt.plot([0,len(df)],[min(cumulative), min(cumulative)],'--k')

    elif plot == True and include_negatives==False:
        cumulative = np.append(0, np.cumsum(df_order))
        plt.figure(figsize = (16,8))
        plt.plot([0,len(df)], cumulative[[0,len(df)]])
        plt.plot([len(df),len(df)],[min(cumulative), max(cumulative)],'--k')
        plt.plot([0,len(df)],[min(cumulative), min(cumulative)],'--k')

    Lorenz = plt.plot(cumulative, 'blue')
    plt.ylabel('Cumulative')
    plt.xlabel('Month')
    plt.legend(Lorenz, ['Lorenz Curve'])
    plt.text(x=0,y=cum.max()-int(cum.max()*.15),s='Gini coefficient: ' + str(round(gini,3)), fontsize=16)
    plt.show()
    
    return gini
    
'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
dfa = pd.read_excel(url, sheet_name = 'M0', index_col = 0)
dfa.index = pd.to_datetime(dfa.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
dfb = dfa.diff(1).dropna()               # take first difference

dfc = pd.DataFrame(np.array([1, 1, 2, 2,25,250,3000]))
dfd = pd.DataFrame(np.array([-1,-48,-680,98,487,8000]))

Gini(df=dfa,include_negatives=False, plot=True)
Gini(df=dfb,include_negatives=True, plot=True)
'''