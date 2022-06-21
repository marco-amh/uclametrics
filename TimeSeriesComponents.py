# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:07:44 2022

@author: Marco Antonio Martinez Huerta
marco.martinez@ucla.edu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
   
def TimeSeriesComponents(df, lags=0,polynomial_degree=0,polynomial_lags=0,linear_trend=False,quadratic_trend=False,season=False,constant=False):
    df_original = df.copy()
    df1 = df.copy()
    
    N = df1.shape[0]
    M = df1.shape[1]

    if lags > 0:
        df2 = pd.DataFrame() # Support matrix
        for i in range(0,lags+1):
            tmp = df_original.shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df2 = pd.concat([df2,tmp], axis=1)
        df1 = df2
        df1.columns = df1.columns.str.replace("_0", "")
    if polynomial_degree > 0:
        df2  = df_original.iloc[:,1:] 
        poly = PolynomialFeatures(degree=polynomial_degree,include_bias=False, interaction_only=False)
        df_pol = pd.DataFrame(poly.fit_transform(df2))
        pol_names = poly.get_feature_names(df2.columns)
        df_pol.columns = pol_names
        df_pol = df_pol.iloc[:,M:]     
        df_pol.index =   df1.index 
        df1 = pd.concat([df1, df_pol], axis=1)
    if polynomial_lags > 0:
        df2 = pd.DataFrame() # Support matrix
        for i in range(1,polynomial_lags+1):
            tmp = df_pol.shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df2 = pd.concat([df2,tmp], axis=1)
        
        df2.columns = df2.columns.str.replace("_0", "")
        df1 = pd.concat([df1, df2], axis=1)
    if season == True:
        df1['month'] = pd.DatetimeIndex(df1.index).strftime('%b')
        dummy = pd.get_dummies(df1['month'])
        dummy = dummy.drop(columns = [dummy.columns[0]])
        df1= pd.concat([df1, dummy], axis = 1)
        df1 = df1.drop(columns = ['month'])
    if linear_trend == True:
        df1['lin_trend'] = np.arange(0,N)
    if quadratic_trend == True:
        df1['quad_trend'] = np.arange(0,N) ** 2
    if constant == True:
        df1['constant'] = 1
    return  df1.dropna()

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
dfa = pd.read_excel(url, sheet_name = 'ols', index_col = 0)
dfa.index = pd.PeriodIndex(dfa.index, freq = 'M').astype(str) + '-01'
a = TimeSeriesComponents(df=dfa, polynomial_degree=2,polynomial_lags=3,lags=1,linear_trend=True,quadratic_trend=True,season=True,constant=True)
'''