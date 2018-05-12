###########################################################################################################################
# Title: Stepwise Predictors Selection Method
# Course: Math B7800 - Advanced Mathematical Statistics
# Date: May/15/2018, Wednesday
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: The Department of Mathematics, City College of New York, CUNY

# Project Description:
# Consider the Benzene concentration as response Y variable and others as predictors.
# Carry out Stepwise Variable selection method to select an appropriate subset of assumptions.
# Use AIC, Adjusted R^2, and C_p plot to select suitable subset of variable.

# Data Set Information (Air Quiality Data Set from UCI Machine Learning Respository):
# The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded
# in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area,
# at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing
# the longest freely available recordings of on field deployed air quality chemical sensor devices responses.
# Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and
# Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities
# as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required)
# eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.
# This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

# Data Set Sources: https://archive.ics.uci.edu/ml/datasets/Air+quality
###########################################################################################################################
import numpy as np
from numpy import transpose as T
from numpy.linalg import inv
from numpy import matmul as mult
from numpy import dot as dot

begin = 1
end = 5 #9358
data = np.genfromtxt('AirQualityUCI/AirQualityUCI.csv', delimiter=';', usecols = range(2,15), skip_header = 1, dtype=float, loose = True, max_rows = end) #, max_rows = 10
data = np.array(data, dtype=float)

Y = data[begin:end,3]
n = Y.shape[0]
print('n:', n)
# print('Y:', Y)
# Z = data[0:n,10:12]
# Z = data[0:n,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

Z = np.delete(data,3,axis=1)[begin:end,] # axis=1 -- select a column, axis=0 -- select a row.
r = Z.shape[1]
# print('r: ', r)
Z = np.insert(Z, 0, np.ones(n), axis=1)
print('Z:')
print(Z)

def getBetaHat(data, response):
    z = 0; y = 0; beta_hat=0
    z = data
    y = response

    beta_hat = inv(z.T.dot(z)).dot(z.T).dot(y)

    return beta_hat

# print('Beta_hat:')
# print(getBetaHat(Z, Y))

def getProjectionMatrix(data):
    z = 0
    Pz = 0
    z = data
    Pz = z.dot(inv(z.T.dot(z))).dot(z.T)
    return Pz

# print(getProjectionMatrix(Z))

def getPredictedResponse(data, response):
    Pz = 0; z = 0; y = 0; y_hat = 0; y_hat1 = 0

    z = data
    y = response
    Pz = getProjectionMatrix(z)
    beta_hat = getBetaHat(z, y)

    y_hat =  Pz.dot(y)
    y_hat1 = z.dot(beta_hat)
    # print('result1: ')
    # print(y_hat)
    # print('result2: ')
    # print(y_hat1)
    return y_hat

print('Y_hat:')
print(getPredictedResponse(Z, Y))

def getResidualSS(data, response):

    resSS=0;z=0;y=0;
    n=0;eye=0;Pz=0

    z = data
    y = response
    n = y.shape[0]
    eye = np.eye(n)
    Pz = getProjectionMatrix(z)

    resSS = y.T.dot(eye-Pz).dot(y)

    return resSS

print(getResidualSS(Z,Y))


def getUnbiasedResidualSS(data, response, observations, variables):

    z=0;y=0;n=0;r=0;s2=0
    z = data; y = response
    n = observations; r = variables

    residualSS = getResidualSS(z, y)
    s2 = residualSS/(n-r-1)

    return s2

print(getUnbiasedResidualSS(Z, Y, n, r))

































#print('Hello!')
