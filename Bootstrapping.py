# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 18:33:09 2022

@author: marco
"""

import pandas as pd
import numpy as np
import random
import math
import os
from scipy.linalg import pinv as inv
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
os.chdir('C://Users//marco//Desktop//Projects//Bootstrapping')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')


df = pd.read_excel('Data.xlsx', index_col=0)

def bs(df,B):
    for i in range(0,B):
        df.sample(df.shape[0], replace=True, random_state=i)


############## Block Bootstrapping

# Loop for block bootstrapping
df=df.to_numpy()
def block_bs(df,B,k):
    k = k #size of blocks
    B = B #number of bootstraps
    n = df.shape[0] #sample size
    m = df.shape[1]# number of features
    s = math.ceil(n/k) #number of blocks in each bootstraps
    coefs = np.zeros((B,m-1))# Matrix to store the results
   
    for i in range(0,B):
        tmp = np.zeros((s*k, m))
        for j in range(1,s+1):
            t = random.sample(range(k,n+1), 1)[0] #last point of time
            tmp[(j-1)*k:j*k , :] =  df[t-k:t,:] #fill the boots vector with observations in a block  
       # Function
        Y = tmp[:, 0:1].reshape(-1,1)
        X = tmp[:,1:]
        lasso = Lasso(alpha = 0.15, fit_intercept=False)
        lasso.fit(X,Y)
        coefs[i:i+1, :]= lasso.coef_
    return(coefs)


df1 = block_bs(df=df,B=10000000,k=3)
