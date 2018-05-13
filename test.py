import numpy as np

def testRecall():

   a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

   while True:

       rand = np.random.randint(10, size=1)[0]
       inx = rand = np.random.randint(10, size=1)[0]

       print('rand: ', rand)
       # print('a[inx]: ', a[inx])

       if rand == 5:
           return rand

       else:

           print('Sorry, please test it again.')

           return testRecall()


print(testRecall())
