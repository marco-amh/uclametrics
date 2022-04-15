rm(list=ls())  	#Clean variables
cat("\014")	    #Clean console

setwd("C://Users//marco//Desktop//UCLA//Capstone_Project")
library(readxl)


library(fImport)



ar_companion_matrix <- function(beta){
  #check if beta is a matrix
  if (is.matrix(beta) == FALSE){
    stop('error: beta needs to be a matrix')
  }
  # dont include constant
  k = nrow(beta) - 1
  FF <- matrix(0, nrow = k, ncol = k)
  
  #insert identity matrix
  FF[2:k, 1:(k-1)] <- diag(1, nrow = k-1, ncol = k-1)
  
  temp <- t(beta[2:(k+1), 1:1])
  #state space companion form
  #Insert coeffcients along top row
  FF[1:1,1:k] <- temp
  return(FF)
}


df <- read_excel("inflation.xls",sheet = "Sheet1")

X=as.matrix(df[,3:5])
Y=as.matrix(df[,2:2])
h = 24
reps = 5000 # number of Gibbs iterations
burn = 4000 # percent of burn-in iterations

out <- matrix(0, nrow = reps, ncol = ncol(X) + 1)
out1 <- matrix(0, nrow = reps, ncol = h)

t1 <- nrow(Y) #number of observations
b0 = matrix(0,ncol(X),1) #Priors
sigma0 <- diag(ncol(X)) # variance matrix

# priors for sigma2

t0= 1
d0=0.1

# Starting values
B=b0
sigma2=1



gibbs_sampler <- function(X,Y,b0,sigma0,sigma2,theta0,D0,reps,out,out1){
  for (i in 1:reps){
    
    M = solve(solve(sigma0)+as.numeric(1/sigma2)*t(X)%*%X)%*%(solve(sigma0)%*%b0+as.numeric(1/sigma2)*t(X)%*%Y)
    V = solve(solve(sigma0)+as.numeric(1/sigma2)*t(X)%*%X)
    chck=-1
    while (chck < 1){ #check for stability
      
      beta= M+t(rnorm(ncol(X))%*%chol(V))
      b = ar_companion_matrix(B)
      ee <- max(sapply(eigen(b)$values,abs))
      if(ee<=1){
        chck=1
      }
    }
    # compute residuals
    resids <- Y- X%*%B
    T2 = t0 + t1
    D1 = d0 + t(resids) %*% resids
    
    # keeps samples after burn period
    out[i,] <- t(matrix(c(t(B),sigma2)))
    ?rnorm
    #draw from Inverse Gamma
    z0 = rnorm(t1,1)
    z0z0 = t(z0) %*% z0
    sigma2 = D1/z0z0
    
    # compute 2 year forecasts
    yhat = rep(0,h)
    end = as.numeric(length(Y))
    #yhat[1:2] = Y[(end-1):end,]
    cfactor = sqrt(sigma2)
    X_mat = c(1,rep(0,ncol(X)-1))
    
    
    for(m in ncol(X):h){
      for (lag in 1:(ncol(X)-1)){
        #create X matrix with p lags
        X_mat[(lag+1)] = yhat[m-lag]
      }
      # Use X matrix to forecast yhat
      yhat[m] = X_mat %*% B + rnorm(1) * cfactor
    }
    
    out1[i,] <- yhat
  }
  return = list(out,out1)
}
