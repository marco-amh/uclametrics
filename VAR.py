# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:52:01 2022

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

dtafile = 'Data.xlsx'
df = pd.read_excel(dtafile, index_col=0, skiprows=0, na_values=('NE'),sheet_name='ols')

# 1. Generación de las matrices de la SUR representation Y=PI*X+U



