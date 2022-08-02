# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:23:00 2022
Code translated from Matlab to Python by
@author: marco antonio martinez huerta
marco.martinez@ucla.edu

"""
import pandas as pd
import numpy as np
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt

def aggreg(op1,N,sc):
    if op1 == 1:
        c = np.ones((1,sc));
    elif op1 == 2:
        c = np.ones((1,sc))/sc;
    elif op1 == 3:
        c= np.zeros((1,sc));
        c[:,sc-1:sc] = 1;
    elif op1 == 4:
        c = np.zeros((1,sc));
        c[:,0:1] = 1
    else:
        print('IMPROPER TYPE OF TEMPORAL DISAGGREGATION')
    C = np.kron(np.eye(N),c)
    return C

def chowlin(Y,x,ta=2,s=3,case=2,plot=False):
    print('Y: Nx1 -> vector of low frequency data                              \n'
          'x: nxp -> matrix of high frequency indicators (without intercept)   \n'
          'if x="constant", then only intercept will be fitted                 \n'
          'ta=1 -> sum (flow)                                                  \n'
          'ta=2 -> average (index)                                             \n'
          'ta=3 -> last element (stock) - interpolation                        \n'
          'ta=4 -> first element (stock) - interpolation                       \n'
          's= 4 -> annual to quarterly                                         \n'
          's=12 -> annual to monthly                                           \n'
          's= 3 -> quarterly to monthly                                        \n'
          'type=0 -> weighted least squares                                    \n'
          'type=1 -> maximum likelihood')
  
    df_original = Y.copy()
    
    original_index = Y.index
    first_date = datetime.strptime(original_index[0], '%Y-%m-%d')
    hf_start_date = first_date + relativedelta(months=-s+1)
    hf_start_date = pd.to_datetime(hf_start_date).strftime('%Y-%m-%d')
    
    names = Y.columns
    
    df_original.rename(columns={names[0]: str(names[0])+'_interpolated'},inplace=True)
    Y=Y.to_numpy()
    N,M = Y.shape    # Size of low-frequency input
    
    if x == 'constant':
        x = np.ones((N*s,1))
        p = 1
        n,p = x.shape    # Size of p high-frequency inputs (just intercept)
    else:
        n,p = x.shape    # Size of p high-frequency inputs (without intercept)
        # Preparing the X matrix: including an intercept
        e = np.ones((n,1))   
        x = np.c_[e,x] # Expanding the regressor matrix
        p = p+1         # Number of p high-frequency inputs (plus intercept)

    # Generating the aggregation matrix
    C = aggreg(op1=ta,N=N,sc=s)

    # Expanding the aggregation matrix to perform  extrapolation if needed.
    if n > s * N:
        pred = n-s * N           # Number of required extrapolations 
        C = np.c_[C,np.zeros((N,pred))]
    else:
        pred=0
    
    # Temporal aggregation of the indicators
    X = C @ x

# Estimation of optimal innovational parameter by means of a grid search on 
#the objective function: likelihood (type=1) or weighted least squares (type=0)
# Parameters of grid search

    h_lim = 199        # Number of grid points
    inc_rho=0.01       # Step between grid points
    r = np.zeros((1,h_lim)).T
    val = np.zeros((1,h_lim)).T

    r[0] = -0.99
    for h in range(1,h_lim):
        r[h]=r[h-1] + inc_rho

    I = np.eye(n)
    w = I
    L2 = np.zeros((n,n))

    for i in range(1,n):
        L2[i:i+1, i-1:i]= -1  # Auxiliary matrix useful to simplify computations

# Evaluation of the objective function in the grid
    
    for h in range(0,h_lim):
        Aux = I + (r[h]* L2)
        Aux[0,0] = np.sqrt(1-r[h]**2)
        w = inv(Aux.T @ Aux)            # High frequency VCV matrix (without sigma_a)
        W = C @ w @ C.T                   # Low frequency VCV matrix (without sigma_a)
        Wi = inv(W)
        beta = inv(X.T @ Wi @ X) @ (X.T @ Wi @ Y)      # beta estimator
        U = Y - X @ beta                 # Low frequency residuals
        scp = U.T @ Wi @ U                # Weighted least squares
        sigma_a = scp / N              # sigma_a estimator
        # Likelihood function
        l = (-N/2) * np.log(2* np.pi * sigma_a) - (1/2) * np.log(np.linalg.det(W))-(N/2)
        if case == 1:
            val[h]= -scp;   # Objective function = Weighted least squares
        elif case ==2:
            val[h] = l      # Objective function = Likelihood function
# Determination of optimal rho
    hmax = np.argmax(val)
    valmax = np.max(val)
    rho = r[hmax]

    # Final estimation with optimal rho
    Aux = I + (rho * L2)
    Aux[0,0] = np.sqrt(1- rho**2)
    w = pinv(Aux.T @ Aux)           # High frequency VCV matrix (without sigma_a)
    W = C @ w @ C.T                  # Low frequency VCV matrix (without sigma_a)
    Wi = pinv(W)
    beta = pinv(X.T @ Wi @ X) @ (X.T @ Wi@ Y)  # beta estimator
    U = Y - X @ beta                # Low frequency residuals
    scp = U.T @ Wi @ U               # Weighted least squares
    sigma_a = scp / (N-p)         # sigma_a estimator
    L = w @ C.T @ Wi                 # Filtering matrix
    u = L @ U                     # High frequency residuals

    #Temporally disaggregated time series
    y = x @ beta + u

# Information criteria
# Note: p is expanded to include the innovational parameter

    aic = np.log(sigma_a)+ 2 * (p+1)/N
    bic = np.log(sigma_a) + np.log(N) * (p+1) / N

    # VCV matrix of high frequency estimates

    sigma_beta = sigma_a * pinv(X.T @ Wi @ X)

    VCV_y = sigma_a * (np.eye(n)-L @ C) @ w + (x-L @ X) @ sigma_beta @ (x-L @ X).T

    d_y = np.sqrt((np.diag(VCV_y)))   # Std of high freq estimates
    y_li = y - d_y                    # Lower band of high freq estimates
    y_ls = y + d_y                    # Upper band of high freq estimates
    
    
    # High-frequency time series formatted
    dates = pd.DataFrame(pd.period_range(hf_start_date , original_index[-1], freq='M'))

    df1 = pd.concat([dates,pd.DataFrame(y)],axis=1, ignore_index=True)
    df1.set_index(0,drop=True,inplace=True)
    df1.rename(columns={1:names[0]},inplace=True)
    df1.index = df1.index.to_timestamp('s')
    df1.index = pd.to_datetime(df1.index, format = '%Y-%m-%d %H:%M:%S', errors='coerce').strftime('%Y-%m-%d')

    # Simple interpolation of original time series to compare
    df2 = pd.concat([df1,df_original],axis=1)
    df2.interpolate(method='linear',limit=None,inplace=True)
    
    if plot==True:
        plt.figure(figsize=(16,8))
        sns.set_style('ticks')
        line, = plt.plot(df2.index, df2.iloc[:,0:1].values, lw=2, linestyle='-', color='black', label='Chow-Lin')
        line, = plt.plot(df2.index, df2.iloc[:,1:2].values, lw=2, linestyle='-', color='red', label='Interpolated')        
        plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
        sns.despine()
        plt.gca().set(title='', xlabel = 'Fecha', ylabel = '')
        plt.xticks(np.arange(0, len(df2), step=round(len(df2)*.05)), rotation=90)
        plt.show() #plot     
            
    return df2



# PURPOSE: Temporal disaggregation using the Chow-Lin method
# ------------------------------------------------------------
# SYNTAX: res=chowlin(Y,x,ta,s,type);
# ------------------------------------------------------------
# OUTPUT: res: a structure
#           res.meth    ='Chow-Lin';
#           res.ta      = type of disaggregation
#           res.type    = method of estimation
#           res.N       = nobs. of low frequency data
#           res.n       = nobs. of high-frequency data
#           res.pred    = number of extrapolations
#           res.s       = frequency conversion between low and high freq.
#           res.p       = number of regressors (including intercept)
#           res.Y       = low frequency data
#           res.x       = high frequency indicators
#           res.y       = high frequency estimate
#           res.y_dt    = high frequency estimate: standard deviation
#           res.y_lo    = high frequency estimate: sd - sigma
#           res.y_up    = high frequency estimate: sd + sigma
#           res.u       = high frequency residuals
#           res.U       = low frequency residuals
#           res.beta    = estimated model parameters
#           res.beta_sd = estimated model parameters: standard deviation
#           res.beta_t  = estimated model parameters: t ratios
#           res.rho     = innovational parameter
#           res.aic     = Information criterion: AIC
#           res.bic     = Information criterion: BIC
#           res.val     = Objective function used by the estimation method
#           res.r       = grid of innovational parameters used by the estimation method
# ------------------------------------------------------------
# SEE ALSO: litterman, fernandez, td_plot, td_print
# ------------------------------------------------------------
# REFERENCE: Chow, G. and Lin, A.L. (1971) "Best linear unbiased 
# distribution and extrapolation of economic time series by related 
# series", Review of Economic and Statistics, vol. 53, n. 4, p. 372-375.
# Bournay, J. y Laroque, G. (1979) "Reflexions sur la methode d'elaboration 
# des comptes trimestriels", Annales de l'INSEE, n. 36, p. 3-30.
# written by:
# Enrique M. Quilis


