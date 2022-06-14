# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:54:33 2022

@author: marco
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import inv as inv
import seaborn as sns

def Lowess(df, r, plot):
    df1 = df.copy()                  # Copy original data frame
    names = df1.columns[0]           # Variable name
    n = df1.shape[0]                 # No. observations

    x= np.arange(0,n).reshape(-1,1)  # linear trend
    y = df1.iloc[:,:1].to_numpy()    # time series to numpy

    yest = np.zeros((n,1))           # Vector to store local regressions
    r = r                            # Window length for weights

    # Last iterations window
    for i in range(len(x)-r+1):
        W = np.zeros((n,n))         # Weight matrix
        w = np.zeros((n,1))         # Weights for each iteration
        b = x[i:i+r]                # window
        c = b - b[int(np.ceil(r/2)-1):int(np.ceil(r/2))] # Distance (xi - x0) in each window
        d = np.abs(c/np.max(c))     # Scaled distance x0 must be zero
        e = (1-d  **3)**3           # Weighting function 
        w[i:i+r] = e                # Weights 
        np.fill_diagonal(W, w)      # Filling Weight matrix 
        ones = np.ones((n,1))       # Adding constant
        X = np.c_[ones,x ]          # Concatenate ones and linear trend
        quad = np.arange(0,n)**2    # Square polynomial
        X = np.c_[X, quad ]         # Concatenate square polynomial
        betahat = inv(X.T @ W @ X) @ (X.T @ W @ y) # Weighted OLS coefficients
        yest[int(np.ceil(r/2))-1+i] =  (X[int(np.ceil(r/2))-1+i] @ betahat) # Fitted y0

    # First iteration
    for i in range(0, int(np.ceil(r/2))):
        W = np.zeros((n,n))             # Empty Weight matrix
        w = np.zeros((n,1))             # Weights for each iteration
        b = x[:r]                       # Fixed window
        c = b - b[i]                    # Distance (xi - x0) in each window
        d = np.abs(c/np.max(np.abs(c))) # Scaled distance x0 must be zero
        e = (1- d ** 3)** 3             # Weighting function
        w[:r] = e                       # Weights 
        np.fill_diagonal(W, w)          # Filling Weight matrix 
        ones = np.ones((n,1))           # Adding constant
        X = np.c_[ones,x ]              # Concatenate ones and linear trend
        quad = np.arange(0,n)**2        # Square polynomial
        X = np.c_[X, quad ]             # Concatenate square polynomial
        betahat = inv(X.T @ W @ X) @ (X.T @ W @ y) #weighted OLS
        yest[i] =  (X[i] @ betahat)     # Fitted y0

    # Middle regressions
    for i in range(int(np.ceil(r/2)),r):
        W = np.zeros((n,n))             # Empty Weight matrix
        w = np.zeros((n,1))             # Weights for each iteration
        b = x[len(x)-r:]                # Fixed window
        c = b - b[i]                    # Distance (xi - x0) in each window
        d = np.abs(c/np.max(np.abs(c))) # Scaled distance x0 must be zero
        e = (1-d  **3)**3               # Weighting function
        w[len(x)-r:] = e                # Weights 
        np.fill_diagonal(W, w)          # Filling Weight matrix 
        ones = np.ones((n,1))           # Adding constant
        X = np.c_[ones,x ]              # Concatenate ones and linear trend
        quad = np.arange(0,n)**2        # Square polynomial
        X = np.c_[X, quad ]             # Concatenate square polynomial
        betahat = inv(X.T @ W @ X) @ (X.T @ W @ y) # Weighted OLS
        yest[len(x)-r+i] =  (X[len(x)-r+i] @ betahat) # Fitted y0 LOWESS


    df1['fitted'] = yest

    if plot==True:    
       plt.figure(figsize=(16,8))
       sns.set_style('ticks')
       line1, = plt.plot(df1.index,df1[names].values, lw=2, linestyle='-', color='black', label=names)
       line2, = plt.plot(df1.index,df1['fitted'].values, lw=2, linestyle='-', color='red', label='Lowess')
       sns.despine()
       plt.legend(frameon=False, title='', loc='best') #edgecolor='white'
       plt.gca().set(title='Local Linear Regression', xlabel = 'Date', ylabel = names)
       plt.xticks(np.arange(0, len(df1), step=round(len(df1)*.05)), rotation=90)
       plt.show() #plot           


'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df = pd.read_excel(url, sheet_name = 'IGAE_sa', index_col = 0)
df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

Lowess(df=df, r=35, plot=True)

'''

# Lowess
'''
https://towardsdatascience.com/loess-373d43b03564
https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm#tcwf
https://xavierbourretsicotte.github.io/loess.html
https://simplyor.netlify.app/loess-from-scratch-in-python-animation.en-us/
https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35
https://www.geeksforgeeks.org/implementation-of-locally-weighted-linear-regression/




def lowess(x, y, f=2. / 3., itera=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))

    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    h1= np.max((x[:, None] - x[None, :]))
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = np.nan_to_num(w, nan=0.0)
    w = (1 - w ** 3) ** 3
    
    yest = np.zeros(n)
    ones = np.ones((n,1))
    x = np.c_[ones,x ]
    quad = np.arange(0,n)**2
    x = np.c_[x, quad ]
    
    delta = np.ones(n).reshape(-1,1)
    for iteration in range(itera):
        for i in range(n):
            weights = (delta * w[:, i])
            W = np.zeros((n,n))
            np.fill_diagonal(W, weights)
            betahat = inv(x.T @ W @ x) @ (x.T @ W @ y)
            yest[i] =  (x[i] @ betahat)
            x[10]

        #residuals = y - yest
        #s = np.median(np.abs(residuals))
        #delta = np.clip(residuals / (6.0 * s), -1, 1)
        #delta = (1 - delta ** 2) ** 2   

    return yest

df2 = lowess(x=x, y=y, f=2/3, itera=5)
pd.DataFrame(Y).plot()
pd.DataFrame(df2).plot()
pd.DataFrame(yest).plot()

np.diag(weights)
inv(x.T @ W @ x) @ (x.T @ W @ y)







def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y
sd = tricubic(x)

def get_weights(distances, min_range):
    max_distance = np.max(distances[min_range])
    weights = tricubic(distances[min_range] / max_distance)
    return weights

get_weights(distances=h, min_range=23)

def get_min_range(distances, window):
    min_idx = np.argmin(distances)
    n = len(distances)
    if min_idx == 0:
        return np.arange(0, window)
    if min_idx == n-1:
        return np.arange(n - window, n)

    min_range = [min_idx]
    while len(min_range) < window:
        i0 = min_range[0]
        i1 = min_range[-1]
        if i0 == 0:
            min_range.append(i1 + 1)
        elif i1 == n-1:
            min_range.insert(0, i0 - 1)
        elif distances[i0-1] < distances[i1+1]:
            min_range.insert(0, i0 - 1)
        else:
            min_range.append(i1 + 1)
    return np.array(min_range)

distances = np.abs(X - n_x)
    min_range = self.get_min_range(distances, window)
    weights = self.get_weights(distances, min_range)



def estimate(self, x, window, use_matrix=False, degree=1):
    n_x = self.normalize_x(x)
    distances = np.abs(n_xx - n_x)
    min_range = self.get_min_range(distances, window)
    weights = self.get_weights(distances, min_range)

    if use_matrix or degree > 1:
        wm = np.multiply(np.eye(window), weights)
        xm = np.ones((window, degree + 1))

        xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
        for i in range(1, degree + 1):
            xm[:, i] = np.power(self.n_xx[min_range], i)


np.abs((x[:, None] - x[None, :])=
'''