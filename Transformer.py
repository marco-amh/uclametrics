# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:07:44 2022

@author: marco
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

# Only lags, ordered by lags
def lagger_a(p,df):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = df.shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    return df1.dropna()

#x= lagger_a(p=15,df=df)

# Lags and contemporaneous; ordered by variable
def lagger_b(p,df):
    df1 = pd.DataFrame() # Support matrix
    for j in range(0,df.shape[1]):
        for i in range(0,p+1):
            tmp = df.iloc[:,j:j+1].shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
    df1.columns = df1.columns.str.replace("_0", "")
    return df1.dropna()

#x= lagger_b(p=15,df=df)

# Lags and contemporaneous, ordered by lags
def lagger_c(p,df):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = df.shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    return df1.dropna()

#x= lagger_c(p=15,df=df)


# Only Lags; ordered by variable
def lagger_d(p,df):
    df1 = pd.DataFrame() # Support matrix
    for j in range(0,df.shape[1]):
        for i in range(0,p+1):
            tmp = df.iloc[:,j:j+1].shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
    df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    return df1.dropna()

#x= lagger_d(p=15,df=df)

# Keep dependent contemporaneous and lags, and only lags of the covariates
def lagger_e(p,df):
    df  = df.copy()
    df1 = pd.DataFrame() # Support matrix
    for j in range(0,df.shape[1]):
        for i in range(0,p+1):
            tmp = df.iloc[:,j:j+1].shift(i)
            tmp.columns = tmp.columns + '_'+ str(i)
            df1 = pd.concat([df1,tmp], axis=1)
    df1.columns = df1.columns.str.replace("_0", "")
    df1.drop(columns=df.columns[1:], inplace=True)
    return df1.dropna()


def normalize(df):
    X=df.to_numpy()
    muhat = np.mean(X,axis = 0).reshape(1,X.shape[1])
    stdhat = np.std(X,axis = 0).reshape(1,X.shape[1])
    Xtilde = np.nan_to_num((X - muhat)/stdhat)
    return Xtilde

#x = normalize(df=df)

def components(df,poly,pol_degree, cycle,lags,linear,quadratic,season,constant):
    names = df.columns
    df1 = df
    m = df1.shape[1]
    
    if poly == True:
        df2  = df1.iloc[:,1:] 
        poly = PolynomialFeatures(degree=pol_degree,include_bias=False, interaction_only=False)
        df_pol = pd.DataFrame(poly.fit_transform(df2))
        pol_names = poly.get_feature_names(df2.columns)
        df_pol.columns = pol_names
        df_pol = df_pol.iloc[:,m:]     
        df_pol.index =   df1.index 
        df1 = pd.concat([df1, df_pol], axis=1)
        
    if cycle == True:
        df2 = pd.DataFrame() # Support matrix
        for j in range(0,df1.shape[1]):
            for i in range(0,lags+1):
                tmp = df1.iloc[:,j:j+1].shift(i)
                tmp.columns = tmp.columns + '_'+ str(i)
                df2 = pd.concat([df2,tmp], axis=1)
        df2.columns = df2.columns.str.replace("_0", "")
        df1 = df2
    if season == True:
        df1['month'] = pd.DatetimeIndex(df1.index).strftime('%b')
        dummy = pd.get_dummies(df1['month'])
        dummy = dummy.drop(columns = [dummy.columns[0]])
        df1= pd.concat([df1, dummy], axis = 1)
        df1 = df1.drop(columns = ['month'])
    if linear == True:
        df1['trend_l'] = np.arange(0,len(df1))
    if quadratic == True:
        df1['trend_q'] = df1.trend_l ** 2
    if constant == True:
        df1['constant'] = 1
    return  df1.dropna()

#x = components(df=df,poly=True,pol_degree=5, cycle=False, lags=10, season=True, linear=True, quadratic=True, constant=True)

def vna(df,t):
    return df.pct_change(t).dropna()

#x = vna(df=df.iloc[:,0:1], t=12)

def ln(df):
    return np.log(df)

#x = ln(df=df.iloc[:,0:1])

def cc(df,t):
    return np.log(df).diff(t).dropna()

#x = cc(df=df.iloc[:,0:1], t=12)

def d(df,t):
    return df.diff(t).dropna()

#x = d(df=df.iloc[:,0:1], t=2)

#df = pd.read_excel('f_Transformer_data1.xls', index_col=0)

def higher_to_lower(df, method):
    df.replace('N/E',np.nan,inplace=True)
    df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    if method == 'mean':
        df1 = df.groupby(pd.PeriodIndex(df.index, freq = 'M')).mean()
        df1.index = df1.index.astype(str) + '-01'
    if method == 'sum':
        df1 = df.groupby(pd.PeriodIndex(df.index, freq = 'M')).sum()
        df1.index = df1.index.astype(str) + '-01'
    if method == 'last':
        df1 = df.groupby(pd.PeriodIndex(df.index, freq = 'M')).last()
        df1.index = df1.index.astype(str) + '-01'
    if method == 'linear':
        df1= df.interpolate(method='linear', axis=0, limit=None, inplace=False).dropna()
    if method == 'repeat':
        df1= df.fillna(method="ffill").dropna()
    return df1

#df = pd.read_excel('f_Bayesian_OLS_inflation.xls', index_col=0)
#x = higher_to_lower(df, method='linear')

