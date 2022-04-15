# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:38:19 2022

@author: marco
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
import warnings                                  # `do not disturbe` mode
import seaborn as sns
from scipy.linalg import qr, pinv, inv
from scipy.linalg import solve_triangular
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from math import ceil
import numpy as np
from scipy import linalg

warnings.filterwarnings('ignore')
os.chdir('C:/Users/marco/Desktop/Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='holt-winters')

n = df.shape[0]
x = np.arange(0,n).reshape(-1,1)
y =  df.to_numpy()
alpha = 0.05

betahat = inv(x.T @ w @  x) @ (x.T @ w @ y)
    # Fitted
fitted = (X @ betahat)

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
    #h = [       (np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = np.nan_to_num(w, nan=0.0)
    w = (1 - w ** 3) ** 3
    s= np.diagonal(w)
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(itera):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

df2 = lowess(x=X.copy(), y=Y.copy(), f=0.11/3, itera=500)
pd.DataFrame(Y).plot()
pd.DataFrame(df2).plot()




for iteration in range(itera):
    print (interation)













































def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

tricubic(x=X)


def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val), min_val, max_val






#def loess(X, y, alpha, deg, all_x = True, num_points = 100):
'''
    Parameters
    ----------
    X : numpy array 1D
        Explanatory variable.
    y : numpy array 1D
        Response varible.
    alpha : double
        Proportion of the samples to include in local regression.
    deg : int
        Degree of the polynomial to fit. Option 1 or 2 only.
    all_x : boolean, optional
        Include all x points as target. The default is True.
    num_points : int, optional
        Number of points to include if all_x is false. The default is 100.

    Returns
    -------
    y_hat : numpy array 1D
        Y estimations at each focal point.
    x_space : numpy array 1D
        X range used to calculate each estimation of y.

    '''
    
    assert (deg == 1) or (deg == 2), "Deg has to be 1 or 2"
    assert (alpha > 0) and (alpha <=1), "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"
    
    if all_x:
        X_domain = X
    else:
        minX = min(X)
        maxX = max(X)
        X_domain = np.linspace(start=minX, stop=maxX, num=num_points)
        
    n = len(X)
    span = int(np.ceil(alpha * n))
    #y_hat = np.zeros(n)
    #x_space = np.zeros_like(X)
    
    y_hat = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)
    
    for i, val in enumerate(X_domain):
    #for i, val in enumerate(X):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)
        
        Nx = X[ind[:span]]
        Ny = y[ind[:span]]
        
        delx0 = sorted_dist[span-1]
        
        u = distance[ind[:span]] / delx0
        w = (1 - u**3)**3
        
        W = np.diag(w)
        A = np.vander(Nx, N=1+deg)
        
        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = qr(V)
        p = solve_triangular(R, np.matmul(Q.T, Y))
        #p = np.matmul(pinv(R), np.matmul(Q.T, Y))
        #p = np.matmul(pinv(V), Y)
        y_hat[i] = np.polyval(p, val)
        x_space[i] = val
        
    return y_hat, x_space

y_hat2, x_space2 = loess(df1['yt'], df1['trend'], 0.05, 2, all_x = True, num_points = 200)

pd.DataFrame(y_hat2).plot()
pd.DataFrame(x_space2).plot()
