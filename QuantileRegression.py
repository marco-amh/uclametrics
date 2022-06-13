# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:41:51 2022

@author: marco
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from Density import density_plots
from scipy.linalg import inv as inv

# Falta hacer bootstrapping para S.E.

def quant_reg(df=None,const=True,test=1, plot=True,plot_bivariate=True):
    df1=df.copy()
    X = df1.iloc[:,1:]
    y = df1.iloc[:,0:1]
    
    if const== True:
        X['const'] = 1
        first_column = X.pop('const')
        X.insert(0, 'const', first_column)
    else:
        pass
    
    
    Xnames = X.columns
    Xnames = np.insert(Xnames, 0, 'q')
    Ynames = y.columns
    X = X.to_numpy()
    Y = y.to_numpy()
    K = X.shape[1]
    N = X.shape[0]
    
    ols = inv(X.T@X) @ (X.T @ Y)
    
    results = np.ones((99,K+1))*np.nan
    results[:,0:1]=np.arange(0.01,1,0.01).reshape(-1,1)
    j=0
    for i in np.arange(0.01,1,0.01):
        tau = i

    # equality constraints - left hand side
    #1st term: intercepts & data points - positive weights
    #2nd term: intercept & data points - negative weights
    #3rd term: error - positive
    #4th term: error - negative
        A = np.concatenate((X,X*-1,np.identity(N),np.identity(N)*-1 ), axis= 1) #all the equality constraints 

    # equality constraints - right hand side
        b = y.to_numpy() 
        
        #goal function - intercept & data points have 0 weights
    #positive error has tau weight, negative error has 1-tau weight
        c = np.concatenate((np.repeat(0,2*K), tau*np.repeat(1,N), (1-tau)*np.repeat(1,N) ))

    #converting from numpy types to cvxopt matrix

        Am,bm,cm = matrix(A), matrix(b), matrix(c)

    # all variables must be greater than zero
    # adding inequality constraints - left hand side
        n = Am.size[1]
        G = matrix(0.0, (n,n))
        G[::n+1] = -1.0

    # adding inequality constraints - right hand side (all zeros)
        h = matrix(0.0, (n,1))
        
        #solving the model
        sol = solvers.lp(cm,G,h,Am,bm, solver='glpk')
        
        x = sol['x']

        #both negative and positive components get values above zero, this gets fixed here
        beta = x[0:K] - x[K :2*K]

        results[j:j+1,1:] = beta.T
        j=j+1

    
    results = pd.DataFrame(results)
    #results.set_index(0,drop=True, inplace=True)
    results.rename(columns={0:'q'},inplace=True)
    results.columns= Xnames
    
    fitted = results.iloc[:,1:] * test
    
    if plot==True:
        for j in range(0,K):
            plt.plot(results.q, results.iloc[:,j+1:j+2], color='black', label="Quantile Regression")
            plt.axhline(y=ols[j], color='red', linestyle='-', label="OLS Regression")
    
            plt.ylabel(r"$\beta$")
            plt.xlabel("Quantiles of the conditional " + Xnames[j+1] + " distribution")
            plt.legend()
            plt.show()
        
    

    if plot_bivariate==True:
        for j in range(0,K-1):        
            x = np.arange(np.min(X[:,j+1:j+2]), X[:,j+1:j+2].max(), .01)
            get_y = lambda a, b: a + b * x

            plt.subplots(figsize=(8, 6))

            for i in range(results.shape[0]):
                y = get_y(results.iloc[i:i+1,1:2].values[0], results.iloc[i:i+1,j+2:j+3].values[0])
                plt.plot(x, y, linestyle="dotted", color="grey")

            y = get_y(ols[0:1], ols[j+1:j+2])

            plt.plot(x, y.reshape(-1,1), color="red", label="OLS")
            plt.scatter( X[:,j+1:j+2].reshape(-1,1),Y.reshape(-1,1), alpha=0.2)


            plt.xlabel(Xnames[j+2], fontsize=16)
            plt.ylabel(Ynames[0], fontsize=16)
            plt.show()
        
        
    return results, fitted

'''
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df=pd.read_csv(r"http://freakonometrics.free.fr/rent98_00.txt",sep=r'\,|\t', engine='python')
df = pd.read_excel(url, sheet_name='ols', index_col=0)
a,b = quant_reg(df=df[['b1','b2']],const=True, test=[1,3.2], plot=True, plot_bivariate=True)
density_plots(df=b.iloc[:,0:1].to_numpy())
'''
# References:
#https://www.geeksforgeeks.org/visualizing-the-bivariate-gaussian-distribution-in-python/   
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html
#https://www.geeksforgeeks.org/how-to-perform-quantile-regression-in-python/
#https://towardsdatascience.com/quantile-regression-ff2343c4a03
#https://scikit-learn.org/stable/modules/density.html
#https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html
#https://blogs.sas.com/content/iml/2021/03/08/conditional-distribution-response.html
#https://blogs.sas.com/content/iml/2013/04/17/quantile-regression-vs-binning.html
#https://www.lexjansen.com/wuss/2017/173_Final_Paper_PDF.pdf
#https://support.sas.com/resources/papers/proceedings17/SAS0525-2017.pdf
#https://medium.com/the-artificial-impostor/quantile-regression-part-1-e25bdd8d9d43
#    
#https://freakonometrics.hypotheses.org/59875

