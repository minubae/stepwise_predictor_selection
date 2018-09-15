###########################################################################################################################
# Title: Stepwise Predictors Selection Method
# Course: Advanced Mathematical Statistics II, Spring 2018
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: The Department of Mathematics, CUNY

# Project Description:
# Consider the Benzene concentration as response Y variable and others as predictors.
# Carry out Stepwise Variable selection method to select an appropriate subset of assumptions.
# Use AIC, Adjusted R^2, and C_p plot to select suitable subset of variable.

# Data Set Information:
# The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded
# in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area,
# at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing
# the longest freely available recordings of on field deployed air quality chemical sensor devices responses.
# Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and
# Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities
# as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008
# (citation required)eventually affecting sensors concentration estimation capabilities.
# Missing values are tagged with -200 value.
# This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

# Data Set Sources: https://archive.ics.uci.edu/ml/datasets/Air+quality
###########################################################################################################################
import numpy as np
from numpy import transpose as T
from scipy.stats import f
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt

begin = 1
end_row = 9358
# end_row = 30
###################################################################################
# Import data from a CSV file: 'AirQualityUCI/AirQualityUCI.csv'
###################################################################################

end_col = 15
# end_col = 14

csv_url = 'AirQualityUCI/AirQualityUCI.csv'
data = np.genfromtxt(csv_url, delimiter=';', usecols = range(2,end_col), skip_header = 1, dtype=float, max_rows = end_row)
data = np.array(data, dtype=float)

# Get the Benzene concentration as a Response vector Y from the data
Y = data[begin:end_row,3]

# Get the number of observations
n = Y.shape[0]
# print('n:', n)

# Get a Data Matrix Z from the data
Z = np.delete(data,3,axis=1)[begin:end_row,] # axis=1 -- select a column, axis=0 -- select a row.

# Insert one vector into the data matrix Z
Z = np.insert(Z, 0, np.ones(n), axis=1)
Z = Z.astype(float)
# Get the number of variables in the data matrix Z
r = Z.shape[1]-1
# print('Z:')
# print(Z)
# print('r: ', r)
# print('Original Data:')
# print(data.astype(int))
###################################################################################
# Function for computing Beta_hat
###################################################################################
def getBetaHat(data, response):

    # Initialize variables
    z=[]; y=[]; beta_hat=[]

    # Set variables
    z = data; y = response

    # Compute the beta_hat
    beta_hat = inv(z.T.dot(z)).dot(z.T).dot(y)

    return beta_hat

# print('Beta_hat:')
# print(getBetaHat(Z, Y))

def isInvertible(data):
    z=0; result=0
    z = data
    result = z.shape[0] == z.shape[1] and np.linalg.matrix_rank(z) == z.shape[0]
    return result


###################################################################################
# Function for computing Projection Matrix, i.e., Pz
###################################################################################
def getProjectionMatrix(data, where):

    # Initialize variables
    z=[]; Pz=[]
    # Set variables
    z = data.astype(int)

    # w = where
    # print('where: ', w)
    # print('z')
    # print(z)

    # Compute the Projection Matrix Pz
    Pz = z.dot(inv(z.T.dot(z))).dot(z.T)

    return Pz

# print(getProjectionMatrix(Z))

###################################################################################
# Function for computing Mean Projection Matrix, i.e., P1_n
###################################################################################
def getMeanProjectionMatrix(observations):

    # Initialize variables
    n=0; P1=[]

    # Set variables
    n = observations

    # Compute the Mean Projection Matrix P1_n
    P1 = np.ones((n,n))/n

    return P1

# print('P1: ')
# print(getMeanProjectionMatrix(n))

###################################################################################
# Function for computing Predcted Response, i.e., Y_hat
###################################################################################
def getPredictedResponse(data, response):

    # Initialize variables
    Pz=[]; z=[]; y=[]; y_hat=0; y_hat1=0

    # Set variables
    z = data; y = response

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z, 'getPredictedResponse')

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
    z=[]; y=[]; Pz=[]; n=0;eye=0;resSS=0;

    # Set variables
    z = data; y = response; n = observations

    # Compute the n x n identity matrix
    I = np.eye(n)

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z, 'getResidualSS')

    # Compute the Residual Sum of Squares
    resSS = y.T.dot(I-Pz).dot(y)

    return resSS

# print('ResidualSS: ', getResidualSS(Z,Y,n))

###################################################################################
# Function for computing Unbiased Residual Sum of Squares
###################################################################################
def getUnbiasedResidualSS(data, response, observations, variables):

    # Initialize variables
    z=[]; y=[]; n=0; r=0; s2=0

    # Set variables
    z = data; y = response; n = observations; r = variables

    # Get the Residual Sum of Squares
    residualSS = getResidualSS(z, y, n)

    # Compute the Unbiased Residual Sum of Squares, i.e., s^2
    s2 = residualSS/(n-r-1)

    return s2

# print('UnbiasedResidualSS: ', getUnbiasedResidualSS(Z, Y, n, r))

###################################################################################
# Function for computing Regression Sum of Squares
###################################################################################
def getRegressionSS(data, response, observations):

    # Initialize variables
    z=[]; y=[]; n=0; regSS=0
    z = data; y = response; n = observations

    # Get the Projection Matrix Pz
    Pz = getProjectionMatrix(z, 'getRegressionSS')

    # Get the Mean Projection Matrix P1_n
    P1 = getMeanProjectionMatrix(n)

    # Compute the Regression Sum of Squares
    regSS = y.T.dot(Pz-P1).dot(y)
    return regSS

# print('RegressionSS: ', getRegressionSS(Z, Y, n))

###################################################################################
# Function for computing Total Sum of Squares about Mean
###################################################################################
def getTotalMeanSS(data, response, observations):

    # Initialize variables
    z=[]; y=[]; n=0; totSS=0

    # Set variables
    z = data; y = response; n = observations

    # Compute the n x n Identity Matrix
    I = np.eye(n)

    # Get the Mean Projection Matrix
    P1 = getMeanProjectionMatrix(n)

    # Get the Residual Sum of Squares
    resSS = getResidualSS(z,y,n)

    # Get the Regression Sum of Squares
    regSS = getRegressionSS(z,y,n)

    # Compute the Total Sum of Squares about Mean
    # totSS = y.T.dot(I-P1).dot(y)
    totSS = resSS + regSS

    return totSS

# print('totSS: ', getTotalMeanSS(Z,Y,n))


###################################################################################
# Function for computing R2 (Ratio of Regression Sum of Squares)
###################################################################################
def getRatioRegressionSS(data, response, observations):

    # Initialize variables
    z=[]; y=[]; n=0; R2=0

    # Set variables
    z = data; y = response; n = observations

    # Get RegressionSS
    regSS = getRegressionSS(z, y, n)

    # Get Total SS about Mean
    totSS = getTotalMeanSS(z,y,n)

    # Compute the Ration of Regression SS
    R2 = regSS/totSS

    return R2

# print('R2: ', getRatioRegressionSS(Z, Y, n))

###################################################################################
# Function for computing Adjusted R2 (Adjusted Ratio of Regression Sum of Squares)
###################################################################################
def getAdjustedRatioRegressionSS(data, response):

    # Initialize variables
    z=[]; y=[]; n=0; r=0; Adjusted_R2=0

    # Set variables
    z = data; y = response

    r = z.shape[1]-1; n = z.shape[0]

    # Get the Ration of Regression SS, i.e., R2
    R2 = getRatioRegressionSS(z, y, n)

    # Compute the Adjusted R2:
    Adjusted_R2 = 1-(1-R2)*((n-1)/(n-r-1))

    return Adjusted_R2

# print('Adjusted_R2: ', getAdjustedRatioRegressionSS(Z, Y, n, r))

def isPredictorSignificant(data, data1, response, alpha_value):

    # Initialize variables
    z=[]; z1=[]; Pz=[]; Pz1=[]; I=[]
    r=0; n=0; y=0; alpha=0; df1=0; df2=0; p_value=0; c_value=0

    # Set variables
    z = data; z1 = data1; y = response; alpha = alpha_value

    r = z.shape[1]-1
    n = z.shape[0]
    q = r-1

    I = np.eye(n)
    df1 = r-q
    df2 = n-r-1
    print('')
    print('r: ', r)
    print('q: ', q)
    print('n: ', n)
    print('df1: ', df1)
    print('df2: ', df2)

    # Get projection matrices: Pz and Pz1
    Pz = getProjectionMatrix(z, 'isPredictorSignificant')
    # print(Pz)

    if r==1:

        Pz1 = getMeanProjectionMatrix(n)
        # print(Pz1)
    else:

        Pz1 = getProjectionMatrix(z1, 'isPredictorSignificant')

    # Compute F-ratio and p-value of F-ratio on the F distribution
    numerator = y.T.dot(Pz -Pz1).dot(y)/(df1)
    denomenator = y.T.dot(I -Pz).dot(y)/(df2)

    F = numerator/denomenator
    c_value = f.ppf(1-alpha, df1, df2)

    # p_value = f.cdf(F, df1, df2)
    # print('P-value: ', p_value)

    print('F-ratio: ', F)
    print('C-value: ', c_value)
    print('level of alpha: ', 1-alpha)
    print('')

    Fs_vec = []
    Fs_vec.append(F)
    Fs_vec.append(c_value)

    '''
    plt.title('F test with the level of 0.05')
    # plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
    # plt.legend(loc='upper left')
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    x = [1, 2]
    # label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    label = ['F', 'F_{1,n-3}(0.05)']
    plt.xticks(x, label)
    # plt.xlabel('index of predictors', fontsize=12)
    # plt.ylabel('R2 value', fontsize=12)
    # plt.plot(x, R2_vec)
    plt.bar(x, Fs_vec)
    plt.show()
    # '''

    # Hypothesis test: Reject Ho or not
    if F > c_value:
        # Reject the null hypothesis H0. So the predictor is significant
        return True

    return False


###################################################################################
# Compute Akaike's Information Criterion (AIC)
# Select models having the smaller values of AIC
###################################################################################
def getAIC(data, response):

    z=[]; y=[]; n=0; r=0; p=0; AIC=0

    z = data; y = response
    r = z.shape[1]-1; n = z.shape[0]

    p = r+1

    resSS = getResidualSS(z, y, n)
    AIC = n*np.log(resSS/n)+(2*p)

    return AIC

###################################################################################
# Compute C_p value: Select models with minimum C_p
###################################################################################
def getCp(data, data_subset, response):

    z=[]; zi=[]; y=[]; r=0; n=0; p=0
    numerator=0; denomenator=0; Cp=0

    z = data; zi = data_subset; y = response

    r = z.shape[1]-1; n = z.shape[0]
    p = r+1

    numerator = getResidualSS(zi, y, n)
    denomenator = getResidualSS(z, y, n)

    Cp = numerator/denomenator - (n-2*p)

    return Cp

###################################################################################
# Get a Subset of Data Matrix
###################################################################################
def getSubsetDataMatrix(data, col_index):

    z=[]; zi=[]; ones=[]; index=0; n=0

    z = data; index = col_index

    n = z.shape[0]
    ones = np.ones(n)
    zi = z[:, index]

    zi = np.column_stack((ones, zi))

    return zi

###################################################################################
# Compute a predictor having the most contribution to the Regression SS:
###################################################################################
def getMostPredictorToRegSS(data, response):

    p=0; n=0; predictor_index=0
    R2=0; Adj_R2=0; AIC=0; Cp=0

    R2_vec=[]; Adj_R2_vec=[]; AIC_vec=[]
    Cp_vec=[]; index_vec=[]; z=[]; y=[]

    z = data; y = response

    p = z.shape[1]
    n = z.shape[0]

    for i in range(1,p):

        zi = getSubsetDataMatrix(z, i)

        R2 = getRatioRegressionSS(zi, y, n)
        # R2_vec = np.append(R2_vec, R2)
        R2_vec.append(R2)

    R2_max_index = np.argmax(R2_vec)+1

    print('R2 Max index:')
    print(R2_max_index)

    '''
    plt.title('R2 Value of a regression model with one predictor')
    # plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
    # plt.legend(loc='upper left')
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    plt.xticks(x, label)
    plt.xlabel('index of predictors', fontsize=12)
    plt.ylabel('R2 value', fontsize=12)
    # plt.plot(x, R2_vec)
    plt.bar(x, R2_vec)
    plt.show()
    # plt.savefig('figure_01.png')

    '''

    return R2_max_index

###################################################################################
# Compute an Initial Data matrix with a predictor showing the most contribution
# to the Regression Sum of Squares.
###################################################################################
def getInitDataMatrix(data, response, alpha_value):

    z=[]; zi=[]; y=[]
    index=0; alpha=0; test=False

    z = data; y = response; alpha = alpha_value
    index = getMostPredictorToRegSS(z, y)

    zi = getSubsetDataMatrix(z, index)

    test = isPredictorSignificant(zi, zi[:,0], y, alpha)

    if test == True:

        return zi

    else:

        print('sorry, please try it again')
        z = np.delete(z,index,axis=1)

        return getInitDataMatrix(z, y, alpha)

'''
alpha = 0.05
init_Data = getInitDataMatrix(Z, Y, alpha)
print(init_Data.astype(int))
print(Z.astype(int))
'''

def getUpdatedDataMatrix(data, init_data, response, alpha_value):
    z_int=[]; z_temp=[]; z_updated=[]; z_new=[]; z=[]; y=[]; regSS_vec=[]
    index_vec=[]; RegSS_vec = []

    temp=0; max_index=0; alpha=0; check1=False; check2=False; isEqual=False

    z_int = init_data; z = data; y = response; alpha = alpha_value

    n = z_int.shape[0]
    p1 = z_int.shape[1]
    p2 = z.shape[1]

    for i in range(p1):

        for j in range(p2):

            isEqual = np.array_equal(z_int[:,i], z[:,j])

            if isEqual == True:

                index_vec.append(j)

    z_updated = np.delete(z,index_vec,axis=1)

    # print('z_update+++++++:')
    # print(z_updated.astype(int))
    p2_new = z_updated.shape[1]

    print('hehe++++++++: ', p2_new)
    print(z_updated.astype(int))

    if p2_new == 0:

        z_new = z_int

        return z_new

    else:


        for i in range(p2_new):

            # print(z_updated[:,i])
            z_temp = np.insert(z_int, p1, z_updated[:,i], axis=1)
            regSS = getRegressionSS(z_temp, y, n)

            # print('z_temp: ')
            # print(z_temp.astype(int))
            # print('regSS: ',regSS, i)
            RegSS_vec.append(regSS)

            if  regSS > temp:

                temp = regSS
                max_index = i

            z_temp=[]

        print('max_index: ', max_index+1)
        print('Predictor having Max RegSS:')
        print(z_updated[:,max_index].astype(int))
        print('Regression SS: ')
        print(RegSS_vec)

        z1 = np.column_stack((z_int, z_updated[:,max_index]))

        test = isPredictorSignificant(z1, z_int, y, alpha)

        print(z1.astype(int))

        '''
        plt.title('Regression Sum of Squares (Reg SS)')
        # plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
        # plt.legend(loc='upper left')
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        plt.xticks(x, label)
        plt.xlabel('index of predictors', fontsize=12)
        plt.ylabel('Reg SS value', fontsize=12)
        # plt.plot(x, RegSS_vec)
        plt.bar(x, RegSS_vec)
        plt.show()
        # plt.savefig('figure_01.png')
        '''

        if test == True:

            z_new = np.insert(z_int, p1, z_updated[:,max_index], axis=1)

            return z_new

        else:

            print('Find a another predictor.')

            z_updated = np.delete(z_updated, max_index, axis=1)

                # print('Yo, yo,')
            print(z_updated.astype(int))
                # print(z_updated.shape[1])
                #
                # print(z_int.astype(int))

            if z_updated.shape[1] == 0:

                print('I need to fix this point on May 15th, 2018')
                print(z_int.astype(int))

                return z_int

            z = np.column_stack((np.ones(n), z_updated))

                # print('wow:')
                # print(z.astype(int))

            return getUpdatedDataMatrix(z, z_int, y, alpha)

'''
alpha = 0.05
init_data = getInitDataMatrix(Z, Y, alpha)
print('init_data:')
print(init_data.astype(int))
updated_data = getUpdatedDataMatrix(Z, init_data, Y, alpha)
print('Updated_data: ')
print(updated_data.astype(int))
'''

def getPredictorValidation(data, response, alpha_value): #current_data

    z=[]; y=[]; z_update=[]; current=[]; validation=[]
    n=0; p=0; r=0; add=0;leaves=0; test=False
    z = data; y = response; alpha = alpha_value


    # current = current_data

    n = z.shape[0]
    p = z.shape[1]
    r = p-1

    # z_update = z

    print('')
    print('Validation and Current Model: ')
    print(z.astype(int))

    for i in range(1, p):

        print('counting:', i)
        # print(i)
        zi = z[:,i]

        zi = np.column_stack((np.ones(n), zi))
        # print('zi: ')
        # print(zi.astype(int))

        test = isPredictorSignificant(z, zi, y, alpha)

        if test == False:
            print('bye:leave: ', i)
            z_update = np.delete(z, i, axis=1)

            leaves += 1

            if z_update.shape[1] == 1:

                validation.append(z)
                validation.append(add)
                validation.append(leaves)
                validation = np.array(validation)

                return validation

        else:

            print('hi:add', i)
            add += 1

    print('added: ', add)
    print('leaves: ', leaves)
    # print('test: ', test)
    print('')

    validation.append(z)
    validation.append(add)
    validation.append(leaves)
    validation = np.array(validation)

    return validation

# print('Validation: ')
# updated_data = Z[:,[0, 12, 4]]
# print(updated_data.astype(int))
# validated_data = getPredictorValidation(updated_data, Y, 0.05)
# print(validated_data[0].astype(int))
'''
Fs_vec = [2992084.06887, 3.8424531458]
plt.title('F test with the level of 0.05')
x = [1, 2]
label = ['F', 'F_{1,n-3}(0.05)']
plt.xticks(x, label)
# plt.xlabel('index of predictors', fontsize=12)
# plt.ylabel('R2 value', fontsize=12)
# plt.plot(x, R2_vec)
plt.bar(x, Fs_vec)
plt.show()
'''
# isEqual = np.array_equal(z_int[:,i], z[:,j])

def getStepwisePredictors(data, init_data, response, alpha_value): #significant

    alpha=0; add=0; leaves=0; n=0; p=0; r=0
    new_data=[]; validation=[]; updated_data=[]; checked_data=[]; z=[]; y=[]

    z = data; new_data = init_data; y = response; alpha = alpha_value

    n = z.shape[0]
    p = z.shape[1]
    r = p-1

    print('n: ', n)
    print('r: ', r)

    print('new_data+++++')
    print(new_data.astype(int))
    print(z.astype(int))

    new_r = new_data.shape[1]
    print('new_r:', new_r)


    # p = z.shape[1]
    p1 = new_data.shape[1]
    index_vec = []
    # Z = Z.astype(int)

    for i in range(p1):
        for j in range(p):
            isEqual = np.array_equal(new_data[:,i], z[:,j])
            if isEqual == True:
                # print('delete:')
                # print(z[:,j])
                index_vec.append(j)


    print('Which Predictors are the best?')
    print(index_vec)


    updated_data = getUpdatedDataMatrix(z, new_data, y, alpha)

    validation = getPredictorValidation(updated_data, y, alpha)

    checked_data = validation[0]

    up_r = updated_data.shape[1]

    add = validation[1]
    leaves = validation[2]

    if new_r == up_r and leaves == 0:

        print('Which Predictors are the best?')
        print(index_vec)

        return checked_data

    else:

        return getStepwisePredictors(z, checked_data, y, alpha)



# '''
print('Z:')
print(Z.astype(int))

alpha = 0.05
init_data = getInitDataMatrix(Z, Y, alpha)
# print('init_data:')
# print(init_data.astype(int))

updated_model = getStepwisePredictors(Z, init_data, Y, alpha)
print('Stepwise Predictors: ')
print(updated_model.astype(int))

'''
beta_hat = getBetaHat(Z, Y)
# print(beta_hat.astype(float))
'''

'''
f_vec = [353011.317149, 4837774.80285, 8689232.45902, 657637.479477, 3209017.13757, 11766373.7082,
1696902.75433, 11781374.9159, 11782783.5642, 4706110.61658, 6932575.86162]
plt.title('F-values')
# plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
# plt.legend(loc='upper left')
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
label = ['12', '4', '6', '10', '2', '3', '11', '7', '5', '8', '9']
# label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
plt.xticks(x, label)
plt.xlabel('index of predictors', fontsize=12)
plt.ylabel('F-value', fontsize=12)
# plt.plot(x, RegSS_vec)
plt.bar(x, f_vec)
plt.show()
# plt.savefig('figure_01.png'
'''


'''
z1 = Z[:,[0, 6, 3, 7, 5]]
z2 = Z[:,[0, 4, 2, 11, 8]]
z3 = Z[:,[0, 12, 3, 8, 9]]
z4 = Z[:,[0, 4, 2, 7, 5]]
z5 = Z[:,[0, 12, 10, 2, 11]]
z6 = Z[:,[0, 12, 4, 3, 5]]
z7 = Z[:,[0, 10, 2, 11, 8]]
z8 = Z[:,[0, 2, 3, 7, 8]]
z9 = Z[:,[0, 12, 4, 6, 10]]
z10 = Z[:,[0, 12, 4, 6, 2]]
z11 = Z[:,[0, 12, 4, 10, 2]]
z12 = Z[:,[0, 12, 4, 2, 3]]
# AIC_1 = getAIC(z1, Y)
# AIC_2 = getAIC(z2, Y)
# AIC_3 = getAIC(z3, Y)
# AIC_4 = getAIC(z4, Y)
# AIC_5 = getAIC(z5, Y)
# AIC_6 = getAIC(z6, Y)
# AIC_7 = getAIC(z7, Y)
# AIC_8 = getAIC(z8, Y)
# AIC_9 = getAIC(z9, Y)
# AIC_10 = getAIC(z10, Y)
# AIC_11 = getAIC(z11, Y)
# AIC_12 = getAIC(z12, Y)
# print(AIC_1)
# print(AIC_2)
# print(AIC_3)
# print(AIC_4)
# print(AIC_5)
# print(AIC_6)
# print(AIC_7)
# print(AIC_8)
# print(AIC_9)
# print(AIC_10)
# print(AIC_11)
# print(AIC_12)
AIC_vec = [65774.2950079, 44223.5133876, 20348.0528955, 54668.5972225, 21382.7584556, 7027.51549561, 28658.8018197, 56240.6990775, 3632.89935377, 4887.76100594, 6136.96139745, 7184.34811241]
# plt.title('AIC Values')
# # plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
# # plt.legend(loc='upper left')
# # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
# plt.xticks(x, label)
# plt.xlabel('Index of Subset', fontsize=12)
# plt.ylabel('AIC value', fontsize=12)
# # plt.plot(x, RegSS_vec)
# plt.bar(x, AIC_vec)
# plt.show()
# plt.savefig('figure_01.png'
# r2_1 = getAdjustedRatioRegressionSS(z1, Y)
# r2_2 = getAdjustedRatioRegressionSS(z2, Y)
# r2_3 = getAdjustedRatioRegressionSS(z3, Y)
# r2_4 = getAdjustedRatioRegressionSS(z4, Y)
# r2_5 = getAdjustedRatioRegressionSS(z5, Y)
# r2_6 = getAdjustedRatioRegressionSS(z6, Y)
# r2_7 = getAdjustedRatioRegressionSS(z7, Y)
# r2_8 = getAdjustedRatioRegressionSS(z8, Y)
# r2_9 = getAdjustedRatioRegressionSS(z9, Y)
# r2_9 = getAdjustedRatioRegressionSS(z9, Y)
# r2_10 = getAdjustedRatioRegressionSS(z10, Y)
# r2_11 = getAdjustedRatioRegressionSS(z11, Y)
# r2_12 = getAdjustedRatioRegressionSS(z12, Y)
# print(r2_1)
# print(r2_2)
# print(r2_3)
# print(r2_4)
# print(r2_5)
# print(r2_6)
# print(r2_7)
# print(r2_8)
# print(r2_9)
# print(r2_9)
# print(r2_10)
# print(r2_11)
# print(r2_12)
r2_vec = [0.34079337407, 0.934118023918, 0.994864126101, 0.798829975406, 0.994263605242, 0.998763036734, 0.987516157422, 0.76202541212, 0.999139403518, 0.999015892507, 0.998875336113, 0.998742129275]
# plt.title('Adjusted_R2 Values')
# # plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
# # plt.legend(loc='upper left')
# # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
# plt.xticks(x, label)
# plt.xlabel('Index of Subset', fontsize=12)
# plt.ylabel('Adjusted_R2 value', fontsize=12)
# # plt.plot(x, RegSS_vec)
# plt.bar(x, r2_vec)
# plt.show()

# Cp_1 = getCp(Z, z1, Y)
# Cp_2 = getCp(Z, z2, Y)
# Cp_3 = getCp(Z, z3, Y)
# Cp_4 = getCp(Z, z4, Y)
# Cp_5 = getCp(Z, z5, Y)
# Cp_6 = getCp(Z, z6, Y)
# Cp_7 = getCp(Z, z7, Y)
# Cp_8 = getCp(Z, z8, Y)
# Cp_9 = getCp(Z, z9, Y)
# Cp_9 = getCp(Z, z9, Y)
# Cp_10 = getCp(Z, z10, Y)
# Cp_11 = getCp(Z, z11, Y)
# Cp_12 = getCp(Z, z12, Y)

# print(Cp_1)
# print(Cp_2)
# print(Cp_3)
# print(Cp_4)
# print(Cp_5)
# print(Cp_6)
# print(Cp_7)
# print(Cp_8)
# print(Cp_9)
# print(Cp_9)
# print(Cp_10)
# print(Cp_11)
# print(Cp_12)
Cp_vec = [-8499.32300592, -9247.88110878, -9324.52041075, -9077.19720923, -9323.76277367, -9329.43940602, -9315.24996919, -9030.76334865, -9329.91424287, -9329.75841727, -9329.58108665, -9329.41302848]
plt.title('Cp Values')
# plt.suptitle('The Sum of Squares (Covariances)', x=0.514, y=0.96, fontsize=10)
# plt.legend(loc='upper left')
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
plt.xticks(x, label)
plt.xlabel('Index of Subset', fontsize=12)
plt.ylabel('Cp value', fontsize=12)
# plt.plot(x, RegSS_vec)
plt.bar(x, Cp_vec)
plt.show()
'''







# print('Hello')
