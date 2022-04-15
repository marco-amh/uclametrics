# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:23:00 2022

@author: marco
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import t
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import pinv as pinv
from scipy.linalg import inv as inv
os.chdir('C://Users//marco//Desktop//Projects')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

df = pd.read_excel('f_Temporal_disaggregation.xls', index_col=0)

# Y = low frequency data
# x = high frequency indicators
# ta type of disaggregation
# s = frequency conversion between low and high freq.

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

def chowlin(Y,x,ta=2,s=3,case=2):
     
    N,M = Y.shape    # Size of low-frequency input
    n,p = x.shape    # Size of p high-frequency inputs (without intercept)
# Preparing the X matrix: including an intercept
    e = np.ones((n,1))   
    x = np.c_[e,x] # Expanding the regressor matrix
    p = p+1;         # Number of p high-frequency inputs (plus intercept)

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

    h_lim = 199;        # Number of grid points
    inc_rho=0.01;       # Step between grid points
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
        w = pinv(Aux.T @ Aux)            # High frequency VCV matrix (without sigma_a)
        W = C @ w @ C.T                   # Low frequency VCV matrix (without sigma_a)
        Wi = pinv(W)
        beta = pinv(X.T @ Wi @ X) @ (X.T @ Wi @ Y)      # beta estimator
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

    d_y = np.sqrt((np.diag(VCV_y)))   # Std. dev. of high frequency estimates
    y_li = y - d_y           # Lower lim. of high frequency estimates
    y_ls = y + d_y           # Upper lim. of high frequency estimates

    return y

Y = df.iloc[:,0:1].to_numpy()
x = np.ones((Y.shape[0]*3, 1))   
df1 = chowlin(Y=Y,x=x,ta=2,s=3,case=2)



'''
% PURPOSE: Temporal disaggregation using the Chow-Lin method
% ------------------------------------------------------------
% SYNTAX: res=chowlin(Y,x,ta,s,type);
% ------------------------------------------------------------
% OUTPUT: res: a structure
%           res.meth    ='Chow-Lin';
%           res.ta      = type of disaggregation
%           res.type    = method of estimation
%           res.N       = nobs. of low frequency data
%           res.n       = nobs. of high-frequency data
%           res.pred    = number of extrapolations
%           res.s       = frequency conversion between low and high freq.
%           res.p       = number of regressors (including intercept)
%           res.Y       = low frequency data
%           res.x       = high frequency indicators
%           res.y       = high frequency estimate
%           res.y_dt    = high frequency estimate: standard deviation
%           res.y_lo    = high frequency estimate: sd - sigma
%           res.y_up    = high frequency estimate: sd + sigma
%           res.u       = high frequency residuals
%           res.U       = low frequency residuals
%           res.beta    = estimated model parameters
%           res.beta_sd = estimated model parameters: standard deviation
%           res.beta_t  = estimated model parameters: t ratios
%           res.rho     = innovational parameter
%           res.aic     = Information criterion: AIC
%           res.bic     = Information criterion: BIC
%           res.val     = Objective function used by the estimation method
%           res.r       = grid of innovational parameters used by the estimation method
% ------------------------------------------------------------
% INPUT: Y: Nx1 ---> vector of low frequency data
%        x: nxp ---> matrix of high frequency indicators (without intercept)
%        ta: type of disaggregation
%            ta=1 ---> sum (flow)
%            ta=2 ---> average (index)
%            ta=3 ---> last element (stock) ---> interpolation
%            ta=4 ---> first element (stock) ---> interpolation
%        s: number of high frequency data points for each low frequency data points 
%            s= 4 ---> annual to quarterly
%            s=12 ---> annual to monthly
%            s= 3 ---> quarterly to monthly
%        type: estimation method: 
%            type=0 ---> weighted least squares 
%            type=1 ---> maximum likelihood
% ------------------------------------------------------------
% LIBRARY: aggreg
% ------------------------------------------------------------
% SEE ALSO: litterman, fernandez, td_plot, td_print
% ------------------------------------------------------------
% REFERENCE: Chow, G. and Lin, A.L. (1971) "Best linear unbiased 
% distribution and extrapolation of economic time series by related 
% series", Review of Economic and Statistics, vol. 53, n. 4, p. 372-375.
% Bournay, J. y Laroque, G. (1979) "Reflexions sur la methode d'elaboration 
% des comptes trimestriels", Annales de l'INSEE, n. 36, p. 3-30.

% written by:
% Enrique M. Quilis
% Instituto Nacional de Estadistica
% Paseo de la Castellana, 183
% 28046 - Madrid (SPAIN)

t0=clock;

% ------------------------------------------------------------
% Size of the problem

[N,M] = size(Y);    % Size of low-frequency input
[n,p] = size(x);    % Size of p high-frequency inputs (without intercept)

% ------------------------------------------------------------
% Preparing the X matrix: including an intercept

e=ones(n,1);   
x=[e x];       % Expanding the regressor matrix
p=p+1;         % Number of p high-frequency inputs (plus intercept)

% ------------------------------------------------------------
% Generating the aggregation matrix

C = aggreg(ta,N,s);

% -----------------------------------------------------------
% Expanding the aggregation matrix to perform
% extrapolation if needed.

if (n > s * N)
   pred=n-s*N;           % Number of required extrapolations 
   C=[C zeros(N,pred)];
else
   pred=0;
end

% -----------------------------------------------------------
% Temporal aggregation of the indicators

X=C*x;

% -----------------------------------------------------------
% -----------------------------------------------------------
% Estimation of optimal innovational parameter by means of a 
% grid search on the objective function: likelihood (type=1) 
% or weighted least squares (type=0)

% Parameters of grid search

h_lim=199;        % Number of grid points
inc_rho=0.01;     % Step between grid points

r(1:h_lim)=0;
val(1:h_lim)=0;

r(1)=-0.99;
for h=2:h_lim
   r(h)=r(h-1)+inc_rho;
end

I=eye(n); w=I;
LL=zeros(n,n);
for i=2:n
   LL(i,i-1)=-1; % Auxiliary matrix useful to simplify computations
end

% -----------------------------------------------------------
% Evaluation of the objective function in the grid

for h=1:h_lim;
   Aux=I+r(h)*LL;
   Aux(1,1)=sqrt(1-r(h)^2);
   w=inv(Aux'*Aux);            % High frequency VCV matrix (without sigma_a)
   W=C*w*C';                   % Low frequency VCV matrix (without sigma_a)
   Wi=inv(W);
   beta=(X'*Wi*X)\(X'*Wi*Y);   % beta estimator
   U=Y-X*beta;                 % Low frequency residuals
   scp=U'*Wi*U;                % Weighted least squares
   sigma_a=scp/N;              % sigma_a estimator
   % Likelihood function
   l=(-N/2)*log(2*pi*sigma_a)-(1/2)*log(det(W))-(N/2);
   switch type
   case 0
      val(h)=-scp;   % Objective function = Weighted least squares
   case 1
      val(h)=l;      % Objective function = Likelihood function
   end; 
end; % of loop h

% -----------------------------------------------------------
% Determination of optimal rho

[valmax,hmax]=max(val);
rho=r(hmax);

% -----------------------------------------------------------
% Final estimation with optimal rho

Aux=I+rho*LL;
Aux(1,1)=sqrt(1-rho^2);
w=inv(Aux'*Aux);           % High frequency VCV matrix (without sigma_a)
W=C*w*C';                  % Low frequency VCV matrix (without sigma_a)
Wi=inv(W);
beta=(X'*Wi*X)\(X'*Wi*Y);  % beta estimator
U=Y-X*beta;                % Low frequency residuals
scp=U'*Wi*U;               % Weighted least squares
sigma_a=scp/(N-p);         % sigma_a estimator
L=w*C'*Wi;                 % Filtering matrix
u=L*U;                     % High frequency residuals

% -----------------------------------------------------------
% Temporally disaggregated time series

y=x*beta+u;

% -----------------------------------------------------------
% Information criteria
% Note: p is expanded to include the innovational parameter

aic=log(sigma_a)+2*(p+1)/N;
bic=log(sigma_a)+log(N)*(p+1)/N;

% -----------------------------------------------------------
% VCV matrix of high frequency estimates

sigma_beta=sigma_a*inv(X'*Wi*X);

VCV_y=sigma_a*(eye(n)-L*C)*w+(x-L*X)*sigma_beta*(x-L*X)';

d_y=sqrt((diag(VCV_y)));   % Std. dev. of high frequency estimates
y_li=y-d_y;           % Lower lim. of high frequency estimates
y_ls=y+d_y;           % Upper lim. of high frequency estimates

% -----------------------------------------------------------
% -----------------------------------------------------------
% Loading the structure

res.meth='Chow-Lin';

% -----------------------------------------------------------
% Basic parameters 

res.ta        = ta;
res.type      = type;
res.N         = N;
res.n         = n;
res.pred      = pred;
res.s         = s;
res.p         = p;

% -----------------------------------------------------------
% Series

res.Y         = Y;
res.x         = x;
res.y         = y;
res.y_dt      = d_y;
res.y_lo      = y_li;
res.y_up      = y_ls;

% -----------------------------------------------------------
% Residuals

res.u         = u;
res.U         = U;

% -----------------------------------------------------------
% Parameters

res.beta      = beta;
res.beta_sd   = sqrt(diag(sigma_beta));
res.beta_t    = beta./sqrt(diag(sigma_beta));
res.rho       = rho;

% -----------------------------------------------------------
% Information criteria

res.aic       = aic;
res.bic       = bic;

% -----------------------------------------------------------
% Objective function

res.val       = val;
res.r         = r;

% -----------------------------------------------------------
% Elapsed time

res.et        = etime(clock,t0);





'''