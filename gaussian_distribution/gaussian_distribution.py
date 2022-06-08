import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

def gen_cov_matrix( size, rho ):
    matrix = np.ones((size,size))
    for i in range(size):
        for j in range(size):
            matrix[i,j] = rho**(abs(i-j))
    return matrix

def gen_datas( mean, rho, k=10000, n=50):
    dataY = np.random.normal(size=(k, n))
    #print(data0.shape)
    #print(data0)
    #print('mean :', sum(data0.T[0])/10000)
    #print('standard deviation :', math.sqrt(sum((data0.T[0]-(sum(data0.T[0])/10000))**2)/10000))

    cov = gen_cov_matrix( n, rho)
    #print(cov)

    eigval, eigvec = np.linalg.eig(cov)
    #print(eigval)
    #print(eigvec)
    #print(eigvec.shape, eigval.shape)

    diag = np.diag(1/(eigval**0.5))
    whitening = (np.dot(eigvec, diag)).T
    inv_whitening = np.linalg.inv(whitening)
    #print(whitening.shape, inv_whitening.shape)

    dataX = (np.dot(inv_whitening, (dataY.T))).T + mean
    print(dataX)
    print(dataX.shape)
    print('mean :', sum(dataX.T[0])/10000)
    print('standard deviation :', math.sqrt(sum((dataX.T[0]-(sum(dataX.T[0])/10000))**2)/10000))

    return dataX

data0 = gen_datas(0, 0.9, n=50)
data1 = gen_datas(0.5, 0.7, n=50)

np.save('data0.npy', data0)
np.save('data1.npy', data1)


plt.plot(data1.T[0], data1.T[1],'og')
plt.plot(data0.T[0], data0.T[1],'oc')
plt.savefig('datas.png')
