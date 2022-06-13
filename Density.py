# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 11:05:03 2022

@author: marco
"""
# Monte Carlo Simulation OLS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

def density_plots(df):
    n = df.shape[0]
    # Number of bins - histogram
    bins = np.arange(np.mean(df)-(4*np.std(df,ddof=1)),np.mean(df)+(4*np.std(df,ddof=1)),0.0001)
    # Kernel density function and histogram
    kernel = sns.distplot(df,hist=True,kde=True).get_lines()[0].get_data() 
    zz=np.cumsum((kernel[0][1]-kernel[0][0]) * kernel[1]) # kernel cumulative density function
    # Normal probability density function
    plt.plot(bins, 1/(np.std(df,ddof=1)*np.sqrt(2*np.pi)) * np.exp(-(bins-np.mean(df))**2 / (2*np.std(df,ddof=1)**2)),
         linewidth=2, color='r')
    # T-student probability density function
    pdf_fitted = t.pdf(bins,loc=t.fit(df)[1],scale=t.fit(df)[2],df=n)
    yy=np.cumsum(0.0001 * pdf_fitted) # t-student cumulative density function
    
    plt.plot(bins,pdf_fitted)
    plt.show()
    plt.plot(bins,yy)
    plt.title('t-Student - cumulative density function')
    plt.show()
    plt.plot(kernel[0],zz)
    plt.title('Kernel - cumulative density function')
    plt.show()

#s = np.random.normal(64, 60, 1000)
#dens(df=s)

#Note: don't matter that the y-axis range is above 1. To get the pr of 
# a point data, you need to multiply the bin width by the pdf value.

'''
t.ppf(.975,df=360) #qnorm 
t.cdf(1.64,df=360) #cum dist func

# Normal probability density function
#\frac{1}{\sigma * (2\pi)^{1/2}}* e^{-\frac{(30-\mu)^2}/{2*\sigma^2}}
d = 1/(sigma * np.sqrt(2 * np.pi)) *  np.exp( - (1/2)*((bins - mu)/ sigma)**2) # gauss normal beell
#https://en.wikipedia.org/wiki/Normal_distribution

# t-student probabiluity density function
from scipy.special import beta as bt
a = (1 / (np.sqrt(n) * bt((1/2),(n/2)) ))*(1/ (1+(tsq/n))**((n+1)/2))
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.beta.html
#https://en.wikipedia.org/wiki/Student%27s_t-distribution
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html

'''


#https://stackoverflow.com/questions/61881175/normed-histogram-y-axis-larger-than-1
#https://www.tutorialspoint.com/seaborn/seaborn_kernel_density_estimates.htm


# t-test for distributions-check this, its super useful
# https://datascienceplus.com/t-tests/


# Other
#What is the benefit of regression with student-t residuals over OLS regression?
#https://stats.stackexchange.com/questions/561001/what-is-the-benefit-of-regression-with-student-t-residuals-over-ols-regression

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html
