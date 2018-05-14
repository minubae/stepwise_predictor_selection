import numpy as np

# D = np.array([[1, 10, 11, 12],[1, 13, 14, 15],[1, 16, 17, 18], [1, 19, 20, 21]])
Yd = np.array([2, 41, 32, 43, 78, 38, 3])
D = np.array([[1, 10, 40, 12, 84],[1, 13, 23, 15, 40],[1, 16, 20, 18, 59],
              [1, 19, 20, 30, 54], [1, 20, 48, 32, 23], [1, 20, 10, 30, 40], [1, 29, 58, 12, 39]])

def getRand():

    rand = np.random.randint(20, size=1)[0]

    return rand


def testRecall():

   a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

   while True:

       # inx = rand = np.random.randint(10, size=1)[0]
       # print('a[inx]: ', a[inx])

       rand = getRand()
       print('rand: ', rand)

       if rand == 5:
           return rand

       else:

           print('Sorry, please test it again.')

           # return testRecall()
           getRand()

           print('hello')


print(testRecall())
