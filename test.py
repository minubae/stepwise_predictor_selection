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
end = 20 #9358
###################################################################################
# Import data from a CSV file: 'AirQualityUCI/AirQualityUCI.csv'
###################################################################################
data = np.genfromtxt('AirQualityUCI/AirQualityUCI.csv', delimiter=';', usecols = range(2,15), skip_header = 1, dtype=float, loose = True, max_rows = end) #, max_rows = 10
data = np.array(data, dtype=float)

# Get a Response vector Y from the data
Y = data[begin:end,3]

# Get the number of observations
n = Y.shape[0]
print('n:', n)
# print('Y:', Y)
# Z = data[0:n,10:12]
# Z = data[0:n,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# Get a Data Matrix Z from the data
Z = np.delete(data,3,axis=1)[begin:end,] # axis=1 -- select a column, axis=0 -- select a row.

# Get the number of variables in the data matrix Z
r = Z.shape[1]
# print('r: ', r)

# Insert one vector into the data matrix Z
Z = np.insert(Z, 0, np.ones(n), axis=1)
# print('Z:')
# print(Z)

###################################################################################
# Function for computing Beta_hat
###################################################################################
def getBetaHat(data, response):

    # Initialize variables
    z = 0; y = 0; beta_hat=0

    # Set variables
    z = data; y = response

    # Compute the beta_hat
    beta_hat = inv(z.T.dot(z)).dot(z.T).dot(y)

    return beta_hat

# print('Beta_hat:')
# print(getBetaHat(Z, Y))

###################################################################################
# Function for computing Projection Matrix, i.e., Pz
###################################################################################
def getProjectionMatrix(data):

    # Initialize variables
    z = 0; Pz = 0
    # Set variables
    z = data

    # Compute the Projection Matrix Pz
    Pz = z.dot(inv(z.T.dot(z))).dot(z.T)
    return Pz

# print(getProjectionMatrix(Z))

###################################################################################
# Function for computing Mean Projection Matrix, i.e., P1_n
###################################################################################
def getMeanProjectionMatrix(data, observations):

    # Initialize variables
    z = 0; n = 0; P1 = 0

    # Set variables
    z = data; n = observations

    # Compute the Mean Projection Matrix P1_n
    P1 = np.ones((n,n))/n

    return P1

# print('P1: ')
# print(getMeanProjectionMatrix(Z, n))

###################################################################################
# Function for computing Predcted Response, i.e., Y_hat
###################################################################################
def getPredictedResponse(data, response):

    # Initialize variables
    Pz = 0; z = 0; y = 0; y_hat = 0; y_hat1 = 0

    # Set variables
    z = data; y = response

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z)

    # Get the Beta_hat
    beta_hat = getBetaHat(z, y)

    # Compute the Predicted Response y_hat
    y_hat =  Pz.dot(y)
    # y_hat1 = z.dot(beta_hat)

    return y_hat

# print('Y_hat:')
# print(getPredictedResponse(Z, Y))

###################################################################################
# Function for computing Residual Sum of Squares
###################################################################################
def getResidualSS(data, response, observations):

    # Initialize variables
    resSS=0;z=0;y=0;n=0;eye=0;Pz=0

    # Set variables
    z = data; y = response; n = observations

    # Compute the n x n identity matrix
    eye = np.eye(n)

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z)

    # Compute the Residual Sum of Squares
    resSS = y.T.dot(eye-Pz).dot(y)

    return resSS

print('ResidualSS: ', getResidualSS(Z,Y,n))

###################################################################################
# Function for computing Unbiased Residual Sum of Squares
###################################################################################
def getUnbiasedResidualSS(data, response, observations, variables):

    # Initialize variables
    z=0;y=0;n=0;r=0;s2=0

    # Set variables
    z = data; y = response; n = observations; r = variables

    # Get the Residual Sum of Squares
    residualSS = getResidualSS(z, y, n)

    # Compute the Unbiased Residual Sum of Squares, i.e., s^2
    s2 = residualSS/(n-r-1)

    return s2

print('UnbiasedResidualSS: ', getUnbiasedResidualSS(Z, Y, n, r))

###################################################################################
# Function for computing Regression Sum of Squares
###################################################################################
def getRegressionSS(data, response, observations):

    # Initialize variables
    z=0; y=0; n=0; regSS=0
    z = data; y = response; n = observations

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z)

    # Get the Mean Projection Matrix P1_n
    P1 = getMeanProjectionMatrix(z,n)

    # Compute the Regression Sum of Squares
    regSS = y.T.dot(Pz-P1).dot(y)
    return regSS

print('RegressionSS: ', getRegressionSS(Z, Y, n))

###################################################################################
# Function for computing Total Sum of Squares about Mean
###################################################################################
def getTotalMeanSS(data, response, observations):

    # Initialize variables
    z=0; y=0; n=0; totSS=0

    # Set variables
    z = data; y = response; n = observations

    # Compute the n x n Identity Matrix
    eye = np.eye(n)

    # Get the Mean Projection Matrix
    P1 = getMeanProjectionMatrix(z, n)

    # Get the Residual Sum of Squares
    resSS = getResidualSS(z,y,n)

    # Get the Regression Sum of Squares
    regSS = getRegressionSS(z,y,n)

    # Compute the Total Sum of Squares about Mean
    # totSS = y.T.dot(eye-P1).dot(y)
    totSS = resSS + regSS

    return totSS

print('totSS: ', getTotalMeanSS(Z,Y,n))


###################################################################################
# Function for computing R2 (Ratio of Regression Sum of Squares)
###################################################################################
def getRatioRegressionSS(data, response, observations):

    # Initialize variables
    z=0; y=0; n=0; R2=0

    # Set variables
    z = data; y = response; n = observations

    # Get RegressionSS
    regSS = getRegressionSS(z, y, n)

    # Get Total SS about Mean
    totSS = getTotalMeanSS(z,y,n)

    # Compute the Ration of Regression SS
    R2 = regSS/totSS

    return R2

print('R2: ', getRatioRegressionSS(Z, Y, n))


###################################################################################
# Function for computing Adjusted R2 (Adjusted Ratio of Regression Sum of Squares)
###################################################################################
def getAdjustedRatioRegressionSS(data, response, observations, variables):

    # Initialize variables
    z=0; y=0; n=0; r=0; Adjusted_R2=0

    # Set variables
    z = data; y = response; r = variables; n = observations

    # Get the Ration of Regression SS, i.e., R2
    R2 = getRatioRegressionSS(z, y, n)

    # Compute the Adjusted R2:
    Adjusted_R2 = 1-(1-R2)*((n-1)/(n-r-1))

    return Adjusted_R2

print('Adjusted_R2: ', getAdjustedRatioRegressionSS(Z, Y, n, r))




















#print('Hello!')
