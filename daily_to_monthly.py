# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:04:03 2022

@author: marco
"""
import pandas as pd

def daily_to_monthly(df):
    df = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean().reset_index()
    df1 = df.iloc[:,0:1].astype(str) + '-01'
    df.index = pd.to_datetime(df1.iloc[:,0:1].squeeze(1), format = '%Y-%m-%d')
    df = df.drop(df.columns[[0]], axis=1)
    return(df)


df = Puller.Banxico(serie="SF43695", name="Monetary_base_daily", plot=True)
daily_to_monthly(df=df)
