# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:21:38 2022

@author: marco
"""

import pandas as pd
import numpy as np
import os
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from scipy.stats import t
from scipy.stats import chi2
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import f
import math
import random
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


def ols(df, lags, constant=False, block_boots=True, B=1000,k=2,heteroskedasticity='White', reset='2', res_ac_lags=12):
    random.seed(a=430)
    # Coefficients
    Y = df.iloc[:,0:1]
    X = df.iloc[:,1:]
    X_names = X.columns
    Y = Y.to_numpy()
    X = X.to_numpy()
    N = X.shape[0]
    M = X.shape[1]
    betahat = inv(X.T @ X) @ (X.T @ Y)
    # Fitted
    fitted = (X @ betahat)
    # Residuals
    ehat = (Y - X @ betahat)
    # OLS Standard Errors
    #Vhat_ols = ((1/(N-M-1)) * ehat.T @ ehat) * inv(X.T @ X)
    #sdhat_ols = np.diagonal(np.sqrt(Vhat_ols))
    #t_ols = betahat/sdhat_ols.reshape(-1,1)   # t-value OLS SE
    
    Vhat_ols = np.identity(M) *(ehat.T @ ehat)  @ inv(X.T @ X)
    sdhat_ols = np.sqrt(np.diagonal(Vhat_ols)) / np.sqrt(N-M-1)
    t_ols = betahat/sdhat_ols.reshape(-1,1)   # t-value OLS SE
    

    # Robust White Standard Errors
    Sigmahat_rob = ((X * ehat).T @ (X * ehat)) / N #Sigma
    Qhat_rob = inv((X.T @ X) / N)
    Vhat_rob = Qhat_rob @ Sigmahat_rob @ Qhat_rob
    sdhat_rob = np.sqrt(np.diagonal(Vhat_rob)) / np.sqrt(N) # Standard Errors
    t_rob = betahat/sdhat_rob.reshape(-1,1) # t-value Robust SE

    # Confidence Interval
    cil = betahat - 1.96 * sdhat_rob;
    cir = betahat + 1.96 * sdhat_rob
    
    # Block Bootstrapping
    if block_boots is True:
        df_tmp = df.to_numpy().copy()
        k = k #size of blocks
        B = B #number of bootstraps
        n = df.shape[0] #sample size
        m = df.shape[1]# number of features
        s = math.ceil(n/k) #number of blocks in each bootstraps
        coefs_bs = np.zeros((B,m-1))# Matrix to store the results
   
        for i in range(0,B):
            tmp = np.zeros((s*k, m))
            for j in range(1,s+1):
                tn = random.sample(range(k,n+1), 1)[0] #last point of time
                tmp[(j-1)*k:j*k , :] =  df_tmp[tn-k:tn,:] #fill the boots vector with observations in a block  
       # Function
            Y_ = tmp[:, 0:1].reshape(-1,1)
            X_ = tmp[:,1:]
            # Function
            betahat_coefs = inv(X_.T @ X_) @ (X_.T @ Y_)
            coefs_bs[i:i+1, :] = betahat_coefs.ravel()
    
        coefs_bs = pd.DataFrame(coefs_bs)
        coefs_bs.columns= X_names
        sdhat_blockbs = np.std(coefs_bs,ddof=1).values
        
        sdhat_blockbs_q = (np.quantile(coefs_bs, 1/4,axis=0) - np.quantile(coefs_bs, 3/4,axis=0)) / (t.ppf(q = 1/4,df = N) - t.ppf(q = 3/4,df = 250))
        t_bs = betahat/sdhat_blockbs.reshape(-1,1) # t-value HAC SE
        t_bs_q = betahat/sdhat_blockbs_q.reshape(-1,1) # t-value HAC SE
    # Newey West Standard Errors
    if lags is None:
        lags = int(np.ceil(1.2 * float(N) **(1/3)))
    # Covariance of errors
    gamma = np.zeros((lags+1))
    
    for i in range(lags+1):
        gamma[i] = (ehat[:N-i].T @ ehat[i:]) / N
   
    w = 1 - np.arange(0,lags+1)/(lags+1)
    w[0] = 0.5
    s2 = gamma @ (2 * w)
    # Covariance of parameters
    Xe = np.zeros((X.shape[0],X.shape[1]))
    for i in range(N):
        Xe[i] = X[i] * float(ehat[i])
    
    Gamma = np.zeros((lags+1,M,M))
    for i in range(lags+1):
        Gamma[i] = Xe[i:].T @ Xe[:N-i]
    Gamma = Gamma / N
    S = Gamma[0].copy()
    for i in range(1,lags+1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)
        
        
    #Sigmahat_rob = ((X * ehat).T @ (X * ehat)) / N #Sigma
    #Qhat_rob = inv((X.T @ X) / N)
    #Vhat_rob = Qhat_rob @ Sigmahat_rob @ Qhat_rob
    #sdhat_rob = np.sqrt(np.diagonal(Vhat_rob)) / np.sqrt(N) # Standard Errors
    #t_rob = betahat/sdhat_rob.reshape(-1,1)
        
    
    Qhat_hac = inv((X.T @ X) / N)
    Vhat_hac = Qhat_hac @ S @ Qhat_hac
    sdhat_hac = np.sqrt(np.diagonal(Vhat_hac)) / np.sqrt(N)
    t_hac = betahat/sdhat_hac.reshape(-1,1) # t-value HAC SE

    # Confidence Interval
    cil = betahat - 1.96 * sdhat_hac;
    cir = betahat + 1.96 * sdhat_hac
    
   
    # Wald Test with OLS estimates
    # Restriction matrix
    
    if constant == True:
        R = np.eye(M)
        R = R[1:,:]
    else:
        R = np.eye((M))

    q_ = R.shape[0]

    Wald_ols = (R @ betahat ).T @ inv(ehat.T @ ehat * (R  @ inv(X.T @ X) @R.T)) @ (R @ betahat )
    Wald_rob = (R @ betahat ).T @ inv( R @ Qhat_rob @ Sigmahat_rob @ Qhat_rob @R.T ) @ (R @ betahat )
    Wald_hac = (R @ betahat ).T @ inv( R @ Qhat_hac @ S @ Qhat_hac @R.T ) @ (R @ betahat )

    Wald_critical_Chi = chi2.ppf(0.95, q_, N)

    p_value_Wald_ols =  1 - chi2.cdf(Wald_ols,q_,N)
    p_value_Wald_rob =  1 - chi2.cdf(Wald_rob,q_,N)
    p_value_Wald_hac =  1 - chi2.cdf(Wald_hac,q_,N)


    Wald_F_ols = Wald_ols / q_
    Wald_F_rob = Wald_rob / q_
    Wald_F_hac = Wald_hac / q_


    Wald_critical_F = f.ppf(0.95, q_, N)

    p_value_Wald_F_ols =  1 - f.cdf(Wald_F_ols,q_, N)
    p_value_Wald_F_rob =  1 - f.cdf(Wald_F_rob,q_, N)
    p_value_Wald_F_hac =  1 - f.cdf(Wald_F_hac,q_, N)
    
    print('Wald Test \n'
          'Ho: b0 =b1=...=bm=0 , Coefficients are zero.\n'
          'Ha: Coefficients are different from zero.\n'
          'We reject Ho if p-value < 0.05 \n'
          'Wald OLS calculated = ' + str(round(Wald_ols[0][0],5)) + '\n' + 
          'Wald White calculated = ' + str(round(Wald_rob[0][0],5)) + '\n' + 
          'Wald HAC calculated = ' + str(round(Wald_hac[0][0],5)) + '\n' + 
          'Wald critical value = ' + str(round(Wald_critical_Chi,5)) +'\n'+
          'Wald OLS p-value = ' + str(round(p_value_Wald_ols[0][0],5))+'\n' +
          'Wald White p-value = ' + str(round(p_value_Wald_rob[0][0],5))+'\n'+
          'Wald HAC p-value = ' + str(round(p_value_Wald_hac[0][0],5))+'\n' )
    
    
    
    # If you want to put restrictions, then fill the following:
    #R[0]= 1
    #R[1]= 1
    #R[2]= 1
    
    
    # OLS White SE
    #np.sqrt(C @ Vhat_rob @ C.T) / np.sqrt(N)
    # OLS SE
    #np.sqrt(C @ Vhat_ols @ C.T) 

    # Residuals
    #Plot residuals
   
    ehat = pd.DataFrame(ehat)
    ehat.plot()
    
    # Histogram
    sm.qqplot(ehat, line ='45', fit=True)
    plt.show()
    sns.distplot(ehat, hist=True, kde=True, bins='auto', color = 'darkblue', 
             hist_kws={'edgecolor':'black'},  kde_kws={'linewidth': 4})
    plt.show()
    
    # Jarque-Bera test for normality
    # Skewness
    skew_1_a = (ehat-np.mean(ehat))**3
    skew_2_a = (np.sum(skew_1_a))/N
    
    skew_1_b = (ehat-np.mean(ehat))**2
    skew_2_b = (np.sum(skew_1_b))/N
    skew_3_b = skew_2_b**(3/2)

    skew_res = skew_2_a/skew_3_b

    # Kurtosis
    kurt_1_a = (ehat-np.mean(ehat))**4
    kurt_2_a = (np.sum(kurt_1_a))/N

    kurt_1_b = (ehat-np.mean(ehat))**2
    kurt_2_b = (np.sum(kurt_1_b))/N
    kurt_3_b = kurt_2_b**(2)

    kurt_res = kurt_2_a/kurt_3_b

    JB_calc_adj = ((N-M)/6) * ((skew_res ** 2) + ((1/4)*(kurt_res-3) ** (2)))[0]
    JB_calc = (N/6) * ((skew_res ** 2) + ((1/4)*(kurt_res-3) ** (2)))[0]
    Chi_critical_JB = chi2.ppf(0.95,2)
    p_value_JB =  1 - chi2.cdf(JB_calc,2)
    
    print('Jarque-Bera Test \n'
          'Ho: JB = 0 , Residuals are Normally distributed.\n'
          'Ha: Residuals do not follow a normal distribution.\n'
          'We reject Ho if p-value < 0.05 \n'
          'JB calculated = ' + str(round(JB_calc,5)) + '\n' + 
          'JB critical value = ' + str(round(Chi_critical_JB,5)) +'\n'+
          'JB p-value = ' + str(round(p_value_JB,5))+'\n')
     


    # Table of results
    results=pd.DataFrame(betahat)
    results.rename(columns={0:'Coef'}, inplace=True)
    results['SE_OLS']       = sdhat_ols
    results['SE_White']     = sdhat_rob
    results['SE_HAC']       = sdhat_hac
    results['SE_BlockBS']   = sdhat_blockbs
    results['SE_BlockBS_q'] = sdhat_blockbs_q
    
    results['t_value_OLS']   = t_ols
    results['t_value_White'] = t_rob
    results['t_value_HAC']   = t_hac
    results['t_value_BS']    = t_bs
    results['t_value_BS_q']  = t_bs_q
    
    results['t_critical'] = t.ppf(q = 1-(0.05/2),df = N)
    
    results['p_value_OLS']       = 2 * (1 - t.cdf(abs(t_ols), N))
    results['p_value_White']     = 2 * (1 - t.cdf(abs(t_rob), N))
    results['p_value_HAC']       = 2 * (1 - t.cdf(abs(t_hac), N))
    results['p_value_BlockBS']   = 2 * (1 - t.cdf(abs(t_bs), N))
    results['p_value_BlockBS_q'] = 2 * (1 - t.cdf(abs(t_bs_q), N))
    

    # R2, centered or uncentered
    if constant == True:
        SSR = (fitted-np.mean(Y)).T @ (fitted-np.mean(Y))
        SSE = ehat.T @ ehat
        SST =  (Y-np.mean(Y)).T @ (Y-np.mean(Y))
        R2 = SSE / SST
        SSR_var = SSR/(M-1)
        SSE_var = SSE/(N-M)
        SST_var = SST/(N-1)
     
    else:
        SSR = fitted.T @ fitted
        SSE = ehat.T @ ehat
        SST =  Y.T @ Y
        R2 = SSE / SST
        SSR_var = SSR/M
        SSE_var = SSE/(N-M)
        SST_var = SST/(N-1)
        

    SSR_sd = SSR_var ** (1/2)
    SSE_sd = SSE_var ** (1/2)
    SST_sd = SST_var ** (1/2)
    
    R2 = 1 - R2
    R2adj = 1-(SSE_var/SST_var)
    
    
    # F test for variances
    
    F_calc = SSR_var/SSE_var
    
    F_critical = f.ppf(0.95, M-1, N-M)
    p_value_F = 1- f.cdf(F_calc, M-1,N-M)
    
    
    print('F-Test \n'
          'F statistic is var(SSR)/var(SSE) \n'
          'Ho: b_1=b_2=...=0. Coefficients are zero.\n'
          'Ha: Not all slope coefficients are zero.\n'
          'We reject Ho if p-value < 0.05 \n' 
          'F calculated = ' + str(round(F_calc[0][0],5)) + '\n' + 
          'F critical value = ' + str(round(F_critical,5)) +'\n'+
          'F p-value = ' + str(round(p_value_F[0][0],5)))
    
    #Selection criterion
    log_lik = (-N/2) * (1 + np.log(2 * np.pi) + np.log(np.sum(ehat **2 )/N))
    AIC = ((-2 * log_lik)/N) + (2*M/N) # Akaike Information Criteria
    AIC_a = (2 * M) - (2*log_lik) # Akaike Information Criteria
    AIC_b = np.log(SSE / N)+(2 * M/N)
    BIC = ((-2 * log_lik)/N) + (M * np.log(N) / N) # Schwarz Bayesian Information 
    BIC_a = (-2 * log_lik) + (M * np.log(N)) # Schwarz Bayesian Information 
    BIC_b = np.log(SSE / N)+(M * np.log(N))
    HannanQuinn = ((-2 * log_lik)/N) + ((2 * M *np.log(np.log(N))/N)) # Hanna Quinn
    HannanQuinn_a = (-2 * log_lik) + (2 * M * np.log(np.log(N))) # Hanna Quinn
    print('Model selection criteria \n' +
          'R^2 = ' + str(round(R2[0][0],5))              + '\n' +
          'R^2 adjusted = ' + str(round(R2adj[0][0],5))  + '\n' +
          'Log-likelihood = ' + str(round(log_lik[0],5)) + '\n' + 
          'AIC = ' + str(round(AIC[0],5))                + '\n'+
          'Hannan Quinn = ' + str(round(HannanQuinn[0],5)) + '\n' + 
          'BIC = ' + str(round(BIC[0],5))+ '\n' +
          'Residual standard error = ' + str(round(SSE_sd[0][0],5)) + '\n')
    
    # Herteroskedasticity
    # Breusch Pagan Test
    
    Yehat = (ehat**2).to_numpy()
    if heteroskedasticity == 'BP':
        Xehat = df.iloc[:,1:].to_numpy()
    elif heteroskedasticity == 'White':
        Xehat = df.iloc[:,1:]
        poly = PolynomialFeatures(degree=2,include_bias=False, interaction_only=False)
        df_pol = pd.DataFrame(poly.fit_transform(Xehat))
        pol_names = poly.get_feature_names(Xehat.columns)
        df_pol.columns = pol_names
        df_pol.index =   Xehat.index 
        Xehat = df_pol.to_numpy()
        
    
    M_e = Xehat.shape[1]
    N_e = Xehat.shape[0]
    # OLS    
    betahat_e = inv(Xehat.T @ Xehat) @ (Xehat.T @ Yehat)
    # Residuals
    ehat_e = (Yehat - Xehat @ betahat_e)
    #R2 and Breusch Pagan Test
    if constant == True:
        R2_e =    1- ((ehat_e.T @ ehat_e)           / ((Yehat-np.mean(Yehat)).T @ (Yehat-np.mean(Yehat))))
        BP_calc = N_e * R2_e
        Chi_critical_BP = chi2.ppf(0.95,2)
        p_value_BP =  1 - chi2.cdf(BP_calc,M_e-1)
    else:
        R2_e =    1- ((ehat_e.T @ ehat_e)           / ((Yehat.T @ Yehat)))
        BP_calc = N_e * R2_e
        Chi_critical_BP = chi2.ppf(0.95,2)
        p_value_BP =  1 - chi2.cdf(BP_calc,M_e)  
    
    print('Heteroskedasticity Tests \n' 
          'Breusch-Pagan Test \n' 
          'Run e^2 = a0 + a1z1 + ... + vi \n' 
          'BP stat is N*R^2 \n'
          'Ho: a_1=a_2=...=0. The coefficients are zero (Homoscedasticity).\n'
          'Ha: Not all slope coefficients are zero.\n'
          'We reject Ho if p-value < 0.05 and conclude that heteroskedasticity is present\n' 
          'BP calculated = ' + str(round(BP_calc[0][0],5)) + '\n' + 
          'BP critical = ' + str(round(Chi_critical_BP,5)) +'\n'+
          'p-value BP = ' + str(round(p_value_BP[0][0],5)) + '\n')
    
    # Test for misspecification
    Xreset = df.iloc[:,1:]
    if reset == '2':
        Xreset['y2'] = (df.iloc[:,:1]**2).to_numpy()
        Xreset = Xreset.to_numpy()
        J = 1
    elif reset == '3':
        Xreset['y2'] = (df.iloc[:,:1]**2).to_numpy()
        Xreset['y3'] = (df.iloc[:,:1]**3).to_numpy()
        Xreset = Xreset.to_numpy()
        J = 2
    
    M_reset = Xreset.shape[1]
    N_reset = Xreset.shape[0]
    # OLS    
    betahat_reset = inv(Xreset.T @ Xreset) @ (Xreset.T @ Y)
    # Residuals
    ehat_reset = (Y - Xreset @ betahat_reset)
    #R2 and Breusch Pagan Test
    SSE_u = ehat_reset.T @ ehat_reset
    reset_calc = ((SSE-SSE_u)/J) / (SSE_u/(N_reset-M_reset))
    F_critical_reset = f.ppf(0.95,M_reset-1,N_reset-M_reset)
    p_value_reset =  1 - f.cdf(reset_calc, M_reset-1,N_reset-M_reset)  
    
    print('Ramsey Model Misspecification Test \n' 
          'Run y = a0 + a1x1 + ... + b1y^2 + b2y^3+ vi \n' 
          'Reset stat is ((SSEr -SSEu)/J) / (SSEu/(N-M)) \n'
          'Ho: b1=0 and/or b2=0. The coefs of the powers are zero.\n'
          'Ha: b1 and/or b2 are not zero.\n'
          'We reject Ho if p-value < 0.05 and conclude that the model is well specified \n' 
          'Reset calculated = ' + str(round(reset_calc[0][0],5)) + '\n' + 
          'Reset critical = ' + str(round(F_critical_reset,5)) +'\n'+
          'p-value Reset = ' + str(round(p_value_reset[0][0],5)) + '\n')    
    
    # Autocorrelation tests
    # Box-Pierce Q-Statistic
    e2 = sm.tsa.acf(ehat, nlags=res_ac_lags)**2
    ljung = 0
    boxp = 0
    for i in range(1,len(e2)):
        boxp += e2[i]
        ljung += e2[i]/(N-i)

    BoxPierce_calc = N*boxp
    Lung_calc = N*(N+2)*ljung
    Chi_critical_AC = chi2.ppf(0.95,res_ac_lags-M)
    p_value_BoxP =  1 - chi2.cdf(BoxPierce_calc,res_ac_lags-M)  
    p_value_Ljung =  1 - chi2.cdf(Lung_calc,res_ac_lags-M)  
    
    print('Autocorrelation Test \n' 
          'Box-Pierce stat is N sum_i=1^k rho^2i \n'
          'Ljung-Box Q stat is N(N+2) \sum_i=1^k (T-i)^-1 rho^2i \n'
          'Ho: p1=p2=...=pk=0. The AC coefs are zero.\n'
          'Ha: Coefs of AC are not zero.\n'
          'We reject Ho if p-value < 0.05 and conclude that the model residuals show autocorrelation \n' 
          'Box-Pierce calculated = ' + str(round(BoxPierce_calc,5)) + '\n' + 
          'Ljung calculated = ' + str(round(Lung_calc,5)) + '\n' + 
          'Tests critical = ' + str(round(Chi_critical_AC,5)) +'\n'+
          'p-value Box-Pierce = ' + str(round(p_value_BoxP,5)) + '\n'  
          'p-value Ljung = ' + str(round(p_value_Ljung,5)) + '\n')  

          
    return results, fitted, ehat, coefs_bs, Xehat, ehat
    

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')




df1 = ols(df=df, constant=False, lags=0,block_boots=True, B=1000,k=2, heteroskedasticity='White', reset='3', res_ac_lags=12)

#SE_HAC
#0	0.0011977698731288101
#1	0.07691584421385882
#2	0.07571360059081236


np.eye((5)).T @ np.eye((5))
# Fix the wald test, the restriction matrix

import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
model1 = ols(formula = 'y ~ b0 +b1+b2-1', data = df)
result1 = model1.fit()
result1.summary()




'''Estimation of a linear regression with Newey-West covari
y : array_like
The dependent variable (regressand). 1-dimensional with T elements.
X : array_like
The independent variables (regressors). 2-dimensional with sizes T
and K. Should not include a constant.
constant: bool, optional
If true (default) includes model includes a constant.
lags: int or None, optional
If None, the number of lags is set to 1.2*T**(1/3), otherwise the
number of lags used in the covariance estimation is set to the value
provided.
Returns
-------
b : ndarray, shape (K,) or (K+1,)
Parameter estimates. If constant=True, the first value is the
intercept.
vcv : ndarray, shape (K,K) or (K+1,K+1)
Asymptotic covariance matrix of estimated parameters
s2 : float
Asymptotic variance of residuals, computed using Newey-West variance
estimator.
R2 : float
Model R-square
R2bar : float
Adjusted R-square
e : ndarray, shape (T,)
Array containing the model errors
Notes
-----
The Newey-West covariance estimator applies a Bartlett kernel to estimate
the long-run covariance of the scores. Setting lags=0 produces White's
Heteroskedasticity Robust covariance matrix.
See also
--------
np.linalg.lstsq
Example
-------
>>> X = randn(1000,3)
>>> y = randn(1000)
>>> b,vcv,s2,R2,R2bar = olsnw(y, X)
Exclude constant:
    
>>> b,vcv,s2,R2,R2bar = olsnw(y, X, False)
Specify number of lags to use:
>>> b,vcv,s2,R2,R2bar = olsnw(y, X, lags = 4)


Y = df.iloc[:,0:1].to_numpy()
X = df.iloc[:,1:].to_numpy()


def olsnw(Y, X, lags=None):
    T = Y.size
    K = X.shape[1]
    if lags is None:
        lags = int(ceil(1.2 * float(T) **(1/3)))
    # Parameter estimates and errors
    betahat = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
    # Residuals
    ehat = (Y - X @ betahat)
    # Newey West
    if lags is None:
        lags = int(ceil(1.2 * float(T) **(1/3)))
    # Covariance of errors
    gamma = np.zeros((lags+1))
    for i in range(lags+1):
        gamma[i] = (ehat[:T-i].T @ ehat[i:]) / T
   
    w = 1 - arange(0,lags+1)/(lags+1)
    w[0] = 0.5
    s2 = gamma @ (2*w)
    # Covariance of parameters
    Xe = np.zeros((X.shape[0],X.shape[1]))
    for i in range(T):
        Xe[i] = X[i] * float(e[i])
    
    Gamma = np.zeros((lags+1,K,K))
    for i in range(lags+1):
        Gamma[i] = Xe[i:].T @ Xe[:T-i]
        Gamma = Gamma/T
    S = Gamma[0].copy()
    for i in range(1,lags+1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)
    
    XpX = (X.T @ X)/T
    XpXi = inv(XpX)
    vcv = (XpXi @ S @ XpXi)/T

    # R2, centered or uncentered
    if constant:
        R2 = (ehat.T @ ehat) / ((Y-mean(Y)).T @ (Y-mean(Y)))
    else:
        R2 =  (ehat.T @ ehat) / (Y.T @ Y)
    R2bar = 1-R2 * (T-1)/(T-K)
    R2 = 1 - R2
    return b,vcv,s2,R2,R2bar,e

olsnw(Y=Y, X=X, constant=True, lags=None)
'''