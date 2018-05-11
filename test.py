import csv
import numpy as np
max_row = 9358
data = np.genfromtxt('AirQualityUCI/AirQualityUCI.csv', delimiter=';', usecols = range(2,15), skip_header = 1, dtype=float, loose = True, max_rows = max_row) #, max_rows = 10

data = np.array(data)
Y = data[:,3]
# print(Y)

for y in Y:
    print(y)
# for d in data:
#     print(d)
# print('hello')
