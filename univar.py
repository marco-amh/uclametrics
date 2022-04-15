# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 14:58:31 2021

@author: marco
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm
from statsmodels.tsa.x13 import x13_arima_analysis as x13
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

class uni:
    def add(df):
        df.index = pd.DatetimeIndex(df.index)
        res = seasonal_decompose(df, model='additive', extrapolate_trend='freq')
        df1 = pd.concat([df,res.trend, res.seasonal, res.resid], axis=1)
        df1.columns = [df.columns[0],'trend','seas', 'resid']
        res.plot()
        return df1
    
    def mul(df):
        df.index = pd.DatetimeIndex(df.index)
        res = seasonal_decompose(df, model='multiplicative', extrapolate_trend='freq')
        df1 = pd.concat([df,res.trend, res.seasonal, res.resid], axis=1)
        df1.columns = [df.columns[0],'trend','seas', 'resid']
        res.plot()
        return df1
    def adf(df): # Model Diagnostic: Unit Root Tests
        # Dickey Fuller
        adf = adfuller(df.x, autolag='AIC')
        print(f'ADF Statistic: {adf[0]}')
        print(f'p-value: {adf[1]}')
        for key, value in adf[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')
    def kpss(df): # Model Diagnostic: Unit Root Tests
        # KPSS Test - Null Hypothesis : Trend Stationary
        kpss_res = kpss(df, regression='c')
        print('\nKPSS Statistic: %f' % kpss_res[0])
        print('p-value: %f' % kpss_res[1])
        for key, value in kpss_res[3].items():
            print('Critial Values:')
            print(f'   {key}, {value}')
    
    def Kalman_Filter(x,j,p,v,plot):
        df1 = pd.DataFrame(x)
        x = x[x.columns[0]]
                
    # Initial parameters
        fcast = []
        fcast_error = []
        kgain = []
        p_tt = []
        x_tt=[]
        filter_error = []
        n = len(x)
        mu = 0 #mean
        a = j #less than 1
        x0 = x11 = 0 #initial x_0=0. This is for part A in this question
        p11 = p00 = 1 #initial variance state model
        v = v # variance state space model
        w = p # variance state representation

        for i in range(0,n):
            x10 = mu + (x11 * a) # 1st step: Estimation "state x_t" equation. Its equal to y_hat
            p10 = (a**2 * p11 ) + w # 2nd step: variance of the state variable
            fcast.append((a**2) * x10)
            fcast_error.append( x[i] - ((a**2) * x10) )# forecast error
            feta = p10 + v # add variance residual v
            k = p10 * (1/feta)
            kgain.append(p10 * (1/feta)) # Kalman Gain
            x11 = ((1-k) * (x10*a)) + (k * x[i]) # filtered state variable
            p11 = p10 - k * p10 #  variance of state variable
            p_tt.append(p11) #  variance of state variable
            x_tt.append(x11) # store filters state variable
            filter_error.append(x[i] - x11) # residual

        gap = x-x_tt
        #x = x
        df1[str(df1.columns[0])+'_filtered'] = x_tt
        df1[str(df1.columns[0])+'_gap'] = gap
        df1[str(df1.columns[0])+'_variance_state'] = p_tt
        df1[str(df1.columns[0])+'_gain'] = kgain
        df1[str(df1.columns[0])+'_fcast_h1'] = fcast
        print('the kalman gain is: ', str(kgain[-1]))
        if plot == True:
            plt.figure(figsize=(16,8))
            plt.style.use('ggplot')
            line, = plt.plot(x.index,x_tt, lw=2, linestyle='-', color='b')
            line, = plt.plot(x.index,x, lw=2, linestyle='-', color='r')
            plt.gca().set(title='kalman', xlabel = 'Date', ylabel = 'variable')
            plt.xticks(np.arange(0, len(x), step=24), rotation=90)
            plt.show() #plot

        return df1
    
    def autocorr(df): # PACF and ACF
        df = df[df.columns[0]]
        fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
        plot_acf(df, lags=24, ax=axes[0])
        plot_pacf(df, lags=24, ax=axes[1])
        plt.show()
    def BoxPierce(df): # Box Pierce and L-Jung 
        df = df[df.columns[0]]
        return acorr_ljungbox(df, boxpierce=True, return_df = True)
    def decompose_(df): # Modeling seasonal dummies with trends
        Y = df[df.columns[0]]
        df['month'] = pd.DatetimeIndex(df.index).strftime('%b')
        dummy = pd.get_dummies(df['month'])
        df['l_trend'] = np.arange(0,len(Y))
        df['q_trend'] = df.l_trend ** 2
        df = df.drop(columns=['month', df.columns[0]])
        X = pd.concat([df,dummy], axis = 1)
        reg_st = OLS(Y, X).fit()
        pd.concat([Y, reg_st.fittedvalues], axis = 1).plot(kind='line',rot=90, figsize=(16,8),title=df.columns[0], xticks=np.arange(0, len(df), step=24))
        return  reg_st.summary()
    def Gini_Finance(df):
        x = df[df.columns[0]]
        x = x.to_frame().dropna().reset_index()
        x=x.drop(columns=x.columns[0])
        x.rename(columns={x.columns[0]: 'col1' }, inplace=True)
        x = x.sort_values(by='col1', ascending=True)
        n = int(len(x))
        cum_x = np.cumsum(x)
        cum_x1 = cum_x.copy()
        cum_x2 = cum_x.copy()

        cum_x1[cum_x1 < 0] = 0
        cum_x2[cum_x2 > 0] = 0

        slope = np.cumsum(x).max()/int(n)

        area_1 = int(n)*(np.cumsum(x).max())/2

        mtx=[]
        j=1
        for i in range(0,len(x)):
            mtx.append(slope*j-cum_x1[i:i+1])
            j=j+1

        val = pd.concat(mtx)
        lorenz_1 = np.sum(val)
        lorenz_2 = abs(np.sum(cum_x2))
        area_2 = 0-cum_x.min()*len(cum_x)

        gini=(lorenz_1+lorenz_2)/(area_1+area_2)
        print("The Gini coefficient is ",gini[0])
        
        ## Draw a Lorenz Curve
        df_order = x.sort_values(by=['col1'])
        cumulative = np.append(0, np.cumsum(df_order))
        # Curve
        plt.figure(figsize = (16,8))
        plt.plot([0,len(x)], cumulative[[0,len(x)]])
        plt.plot([0,len(x)],[0,0], '--k')
        plt.plot([0,0],[min(cumulative), 0],'--k')
        plt.plot([len(x),len(x)],[min(cumulative), max(cumulative)],'--k')
        plt.plot([0,len(x)],[min(cumulative), min(cumulative)],'--k')
        lorenz = plt.plot(cumulative, 'blue')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Month')
        plt.legend(lorenz, ['Lorenz Curve'])
        return gini
    def plot_returns(df):
        x = df[df.columns[0]]
        mu = x.mean()
        sigma = x.std()
        n, bins, patches = plt.hist(x = x, bins='auto', 
                                    color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.ylabel(None)
        plt.title(str(df.columns[0]) + ' Returns Histogram' )
        maxfreq = n.max()
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if (maxfreq % 10 > 0) else maxfreq + 10)
    def skew(df):
        x = df[df.columns[0]]
        n = len(x)
        mean = x.mean()
        std = np.sqrt(np.sum(np.square(x))/(n-1))
        skewness = np.mean(np.power((x - mean)/std, 3))
        return skewness
    def sharpe_ratio(df, rfr):
        x = df[df.columns[0]]
        excess_returns = x - rfr
        n = len(excess_returns)
        mean = excess_returns.mean()
        std = np.sqrt(np.sum(np.square(excess_returns-mean))/(n-1))
        return mean/std
    def best_arima(df, ex):
        x = df[df.columns[0]]
        exog = df.e
        if ex == True:
            results = pm.auto_arima(x, exogenous = exog, error_action='ignore', trace=1)
        else:
            results = pm.auto_arima(x, error_action='ignore', trace=1)
        return results
    def arima(df, ex, param, s_param=[0,0,0,0]):
        x = df[df.columns[0]]
        exog = df.e
        if ex == True:
            arima_res = ARIMA(x, order=(param), seasonal_order=(s_param), exog= exog).fit()
        else:
            arima_res = ARIMA(x, order=(param), seasonal_order=(s_param)).fit()
        arima_fit = arima_res.fittedvalues
        arima_res.summary()
        arima_res.resid.plot()
        arima_res.plot_diagnostics(figsize=(15,5))
        return arima_res, arima_fit
    def x13arima(df, kind='x12'):
        if kind=='x12':
            df = x13(endog = df, outlier=True,print_stdout=True, x12path = os.chdir('C:/Users/marco/Desktop/Projects/WinX13/x13as'), prefer_x13=False)
        elif kind=='x13':
            df = x13(endog = df, outlier=True,print_stdout=True, x12path = os.chdir('C:/Users/marco/Desktop/Projects/WinX13/x13as'), prefer_x13=True)
        
        return df

'''
os.chdir('C:/Users/marco/Desktop/Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='Disaggregation')

df1 = uni.x13arima(df=df, kind='x12')
uni.add(df) # Classical decomposition additive
uni.mul(df) # Classical decomposition multiplicative
uni.Kalman_Filter(x=df,j=0.1,p=0.05,v=0.1,plot=True) # Kalman Filter
uni.autocorr(df) # autocorrelation plots
uni.kpss(df) # Unit Root KPSS
uni.adf_(df) # Unit Root Augmented Dickey Fuller
uni.bp_(df) # Box Pierce and Lung Test for Autocorrelation
uni.decompose(df) # Modeling seasonal dummies with trends
uni.Gini_Finance(df) # Modeling seasonal dummies with trends
uni.plot_returns(df)
uni.skew(df)
uni.sharpe_ratio(df,0.02)
uni.best_arima(df,ex=True)
uni.arima_(df,ex=False, param=[1,1,2], s_param=[0,0,0,0])
uni.arima_(df,ex=False, param=[2,1,3])

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
#https://www.statsmodels.org/stable/tsa.html
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# https://pandas.pydata.org/pandas-docs/version/0.15/generated/pandas.DataFrame.plot.html


#
df3 = ua(df)

df3.resultado = ua(df3.Kalman_Filter(0.1,0.2,0.1)[0])
df3.resultado.add_()

df3.resultado.add_()



# ARIMA Model
arima_mod = ARIMA(df['m'], order=([4, 1, 4]), seasonal_order=(0,0,0,0)).fit()
arima_fit = arima_mod.fittedvalues
arima_mod.summary()
arima_mod.resid.plot()
plt.show()

# Model Diagnostic
arima_mod.plot_diagnostics(figsize=(15,5))

# Forecast
arima_fit.plot_predict(dynamic=False)
arima_mod.predict(2021, 2022)
arima_mod.predict(1,5)



d = {'col1': [1, 2,7,14]}
df = pd.DataFrame(data=d)
'''