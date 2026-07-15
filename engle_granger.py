import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller 

def engle_granger(stock1, stock2):
  X = stock1
  y = stock2

  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  alpha, beta = model.params

  y_hat = alpha + beta * X[:, 1]
  residuals = y - y_hat

  if np.isnan(residuals).any():
    return np.nan
  
  adf_result = adfuller(residuals)

  return adf_result[0], adf_result[1], alpha, beta
  