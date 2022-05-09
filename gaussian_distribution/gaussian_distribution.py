import numpy as np
import math

np.random.seed(1)

def gen_cov_matrix( size, rho ):
    matrix = np.ones((size,size))
    for i in range(size):
        for j in range(size):
            matrix[i,j] = rho**(abs(i-j))
    return matrix

def gen_datas( mean, rho, k=10000, n=50):
    dataY = np.random.normal(size=(k, n))
    #print(data1.shape)
    #print(data1)
    #print('mean :', sum(data1.T[0])/10000)
    #print('standard deviation :', math.sqrt(sum((data1.T[0]-(sum(data1.T[0])/10000))**2)/10000))

    cov1 = gen_cov_matrix( n, rho)
    #print(cov1)

    eigval, eigvec = np.linalg.eig(cov1)
    #print(eigval)
    #print(eigvec)
    #print(eigvec.shape, eigval.shape)

    diag1 = np.diag(1/(eigval**0.5))
    whitening = (np.dot(eigvec, diag1)).T
    inv_whitening = np.linalg.inv(whitening)
    #print(whitening.shape, inv_whitening.shape)

    dataX = (np.dot(inv_whitening, (dataY.T))).T + mean
    print(dataX)
    print(dataX.shape)
    print('mean :', sum(dataX.T[0])/10000)
    print('standard deviation :', math.sqrt(sum((dataX.T[0]-(sum(dataX.T[0])/10000))**2)/10000))

    return dataX

data1 = gen_datas(0, 0.7)
data2 = gen_datas(0.5, 0.9)
