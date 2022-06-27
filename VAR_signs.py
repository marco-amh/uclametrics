# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:35:58 2022

@author: D14371
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv, cholesky # upper traingular
import matplotlib.pyplot as plt
import os
import warnings                             # `do not disturbe` mode

warnings.filterwarnings('ignore')
os.chdir('T://Marco//BOE//')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

# Code base on the BoE bayesian econometrics for central bankers handbook. pgs 55-57

###################################### Lags ##################################
def lagger_a(p,df):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = df.shift(i)
        tmp.columns = tmp.columns + '_' + str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    df1 = df1[df1.columns.drop(list(df1.filter(regex= '_0')))]
    return df1.dropna()
################################### VARs #####################################
url = 'https://github.com/marcovaas/uclametrics/blob/main/Data.xlsx?raw=true'
df0 = pd.read_excel(url, sheet_name = 'boe2_sign', index_col = 0)
df0.index = pd.to_datetime(df0.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

np.random.seed(430)

p = 2 # number of lags
h = 60 # horizon
X = lagger_a(p=p,df=df0) # covariates lags
Y = df0.iloc[p:,:] #endogenous

# Constant
X['cons'] = 1
#first_column = X.pop('cons') # pop this column
#X.insert(0, 'cons', first_column) # to the first position

N = Y.shape[1] # number of endogenous variables
m = X.shape[1] # number of covariate lags
T = X.shape[0] # number of observations

df = pd.concat([Y,X], axis=1, ignore_index=False).dropna()
df.plot(subplots=True)

# Compute standard deviation of each series residual via an ols 
# regression to be used in setting the prior

sigmaP = np.zeros((N,1)) # standar deviation of the S.E. vector
deltaP = np.zeros((N,1)) # AR coeficient


# Note: BOE did this part wrong. We need to adjust line 68, 69, and 71
for i in range(0,N):
    df1 = df0.iloc[:,i:i+1]
    df1['cons'] = 1
    df1['lag'] =  df0.iloc[:,i:i+1].shift(1)
    df1 = df1.iloc[p:,:] 
    Ytmp = df1.iloc[1:,0:1].to_numpy() # df1.iloc[:,0:1].to_numpy()
    Xtmp = df1.iloc[1:,1:].to_numpy()  # df1.iloc[:,1:].to_numpy()
    c = inv(Xtmp.T @ Xtmp) @ Xtmp.T @ Ytmp
    u = Ytmp - Xtmp @ inv(Xtmp.T @ Xtmp) @ Xtmp.T @ Ytmp   # coefficients
    s = (u.T @ u)/(len(u))    # variance of the standard error #(u.T @ u)/(len(u)-2)
    sigmaP[i] = s
    deltaP[i] = c[1]

# Parameters to control the prior
lamdaP = 1         # tightness of the priors on the first lag
tauP = 10 * lamdaP # tightness of the priors on sum coefficients
epsilonP = 1       # tightness of prior on the constant


muP = np.mean(Y.to_numpy(), axis=0).reshape(-1,1)

# Checkpoint: everything equals BOE 4/28/22

def create_dummies(lamda,tau,delta,epsilon,p,mu,sigma,n):
# Creates matrices of dummy observations [...];
#lamda tightness parameter
#tau  prior on sum of coefficients
#delta prior mean for VAR coefficients
# epsilon tigtness of the prior around constant
# mu sample mean of the data
# sigma AR residual variances for the data
# Initialise output (necessary for final concatenation to work when tau=0):

# Get dummy matrices in equation (5) of Banbura et al. 2007:
    if lamda > 0 and epsilon > 0:
        yd1 =np.vstack([np.diag((sigma * delta).ravel())/lamda,
                        np.zeros((n*(p-1),n)),
                        np.diag(sigma.ravel()),
                        np.zeros((1,n))])
            
        jp = np.diag(range(1,p+1))
    
        xd1a = np.hstack([np.kron(jp, np.diag(sigma.ravel())/lamda),np.zeros((n*p,1))])
        xd1b = np.zeros((n,(n*p)+1))
        xd1c = np.hstack([np.zeros((1,n*p))[0], epsilon])
        xd1  = np.vstack([xd1a,xd1b,xd1c])

    else:
        yd1 =np.vstack([np.diag((sigma * delta).ravel())/lamda,
                        np.zeros((n*(p-1),n)),
                        np.diag(sigma.ravel())])
        
        jp = np.diag(range(1,p+1))
        
        xd1a = np.hstack([np.kron(jp, np.diag(sigma.ravel())/lamda),np.zeros((n*p,1))])
        xd1b = np.zeros((n,(n*p)+1))
        xd1  = np.vstack([xd1a,xd1b])
        
    # Get additional dummy matrices - see equation (9) of Banbura et al. 2007:
    if tau>0 and epsilon>0:
        yd2 = np.diag((delta*mu).ravel())/tau
        xd2 = np.hstack([np.kron(np.ones((1,p)),yd2), np.zeros((n,1))])
    else:
        yd2 = np.diag((delta*mu).ravel())/tau
        xd2 = np.kron(np.ones((1,p)),yd2) 
   
 
    y = np.vstack([yd1, yd2])
    x = np.vstack([xd1, xd2])
    
    return y,x


def IWPQ(v,ixpx):
    k = ixpx.shape[0]
    z = np.zeros((v,k))
    mu = np.zeros((k,1))
    for i in range(0,v):
        z[i,:]=(cholesky(ixpx).T @ np.random.normal(0,1, size=(k,1))).T
    out = inv(z.T @ z)
    return out

def getqr(a):

#Returns a modified QR decomposition of matrix a, such that the    %
#diagonal elements of the 'R' matrix are all positive              %
#[Q,R] = QR(A), where A is m-by-n, produces an m-by-n upper triangular
#matrix R and an m-by-m unitary matrix Q (i.e. Q Q'=I) so that A = Q*R.
    q,r = np.linalg.qr(a)

# If diagonal elements of R are negative then multiply the corresponding
# column of Q by -1; Note: the modified Q matrix is still unitary.
    for i in range(0,q.shape[0]):
        if r[i,i]<0:
            q[:,i] = -q[:,i]
    return q

# Return the modified Q matrix


# Pendientes
# Revisar estadística Karina de colocaciones externas
# Apender la estadística externa
# Revisar balances en BMW si reportan 0 en utilidad
# Automatizar nota mensual
# Relación colocaciones netas vs Inversión cuentas nacionales - scatterplot
# Gráfica de tasas de interés del boletín mensual en python
# Ver formato del que refiere Mau
# Actualizr la bitácora de las actividades de la oficina en jupyter

yd,xd = create_dummies(lamdaP,tauP,deltaP,epsilonP,p,muP,sigmaP,N)


#yd and xd are the dummy data. Append this to actual data
Y0 = np.vstack([Y, yd])
X0 = np.vstack([X, xd])
#conditional mean of the VAR coefficients
mstar = (inv(X0.T @ X0) @ X0.T @ Y0).reshape(-1,1,order='F')  #ols on the appended data
xx = X0.T @ X0
ixx = inv(xx @ np.eye(xx.shape[1]))  #inv(X0'X0) to be used later in the Gibbs sampling algorithm
sigma = np.eye(N) #starting value for sigma
REPS = 5000
BURN = 3000
out = np.zeros((REPS-BURN, h,N))


for i in range(0,REPS):
    vstar = np.kron(sigma,ixx)
    beta = mstar + (np.random.normal(0,1, size=(1,N*(N*p+1))) @ cholesky(vstar)).T
    #draw covariance
    e= Y0 - X0 @ beta.reshape(N*p+1,N, order='F')
    scale = e.T @ e
    sigma = IWPQ(T+yd.shape[0], inv(scale))

    if i>=BURN:
        
        #impose sign restrictions
        chck=-1
        while chck < 0:
            K = np.random.normal(0,1,size=(N,N))
            Q = getqr(K)
            A0hat = cholesky(sigma)
            A0hat1 = Q @ A0hat  #candidate draw
        #check signs
            e1 = A0hat1[0,0] > 0 # Response of R
            e2 = A0hat1[0,1] < 0 # Response of Y
            e3 = A0hat1[0,2] < 0 # Response of Inflation
            e4 = A0hat1[0,3] < 0 # Response of consumption
            e5 = A0hat1[0,4] > 0 # Response of U
            e6 = A0hat1[0,5] < 0 # Response of investment
            e7 = A0hat1[0,7] < 0 # response of money
            
            if sum([e1,e2,e3,e4,e5,e6,e7])==7:
                chck=10
            else:
            #check signs but reverse them
                e1 = -A0hat1[0,0] > 0 # Response of R
                e2 = -A0hat1[0,1] < 0 # Response of Y
                e3 = -A0hat1[0,2] < 0 # Response of Inflation 
                e4 = -A0hat1[0,3] < 0 # Response of consumption
                e5 = -A0hat1[0,4] > 0 # Response of U
                e6 = -A0hat1[0,5] < 0 # Response of investment
                e7 = -A0hat1[0,7] < 0 # response of money
            
                if sum([e1,e2,e3,e4,e5,e6,e7])==7:
                    A0hat1[0:1,0:N] = -A0hat1[0:1,0:N]
                    chck = 10
            
            vhat = np.zeros((h,N))
            vhat[p,0] = 1 #  shock to the Federal Funds rate
            yhat = np.zeros((h,N))
            for t in range(p,h):
                yhat[t,:] = np.hstack([yhat[t-1:t,:][0],yhat[t-2:t-1,:][0],0]) @ beta.reshape(N*p+1,N, order='F') + vhat[t,:] @ A0hat1

       
            out[i-BURN,:,:] = yhat
            
out = out[:,p:,:]


for i in np.arange(out.shape[2]):
    g_x = np.arange(out.shape[1])
    g_lo = np.quantile(out[:,:,i], .16, axis=0).ravel()# one s.d. low 
    g_me = np.quantile(out[:,:,i], .50, axis=0).ravel() # median
    g_hi = np.quantile(out[:,:,i], .84, axis=0).ravel()# one s.d. high
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    #colors = plt.cm.Reds(np.linspace(0.3, 0.8, 4))
    ax.fill_between(g_x, g_lo,g_hi, facecolor='#CBC3E3', interpolate=False)
    ax.plot(g_x, g_me, color='#52307c', lw=2, label='Median')
    ax.axhline(y = 0, color = 'red', linestyle = '--')
    ax.set_xticks(g_x)
    #ax.set_xticklabels('df['Time']')
    plt.show()


# Notes:
# T is traspose and is ' in matlab
# inv is the inverse or inv in matlab
# @ is matrix multiplication or * in matlab