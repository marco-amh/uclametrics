# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:35:55 2022

@author: Marco Antonio Martínez Huerta
marco.martinez@ucla.edu
"""
import pandas as pd
import numpy as np

def LowerFreq(df, method, freq):
    df1 = df.copy()
    df1.replace('N/E',np.nan,inplace=True)
    df1.index = pd.to_datetime(df1.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
    if method == 'mean':
        df1 = df1.groupby(pd.PeriodIndex(df1.index, freq = freq)).mean()
    elif method == 'sum':
        df1 = df1.groupby(pd.PeriodIndex(df1.index, freq = freq)).sum()
    elif method == 'last':
        df1 = df1.groupby(pd.PeriodIndex(df1.index, freq = freq)).last()
    elif method == 'linear':
        df1= df1.interpolate(method='linear', axis=0, limit=None, inplace=False).dropna()
    else:
        raise 'Error'
    if freq == 'M':
        df1.index = df1.index.astype(str) + '-01'
    elif freq == 'Q':
        df1.index = pd.PeriodIndex(df1.index, freq = 'M').astype(str) + '-01'
    elif freq == 'A':
        df1.index = df1.index.astype(str) + '-12-01'
    else:
        pass
    return df1


'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name='Transformer', index_col=0)

a = LowerFreq(df, method='mean', freq='M')
b = LowerFreq(df, method='sum', freq='Q')
c = LowerFreq(df, method='last', freq='A')
d = LowerFreq(df, method='linear', freq='M')
'''

