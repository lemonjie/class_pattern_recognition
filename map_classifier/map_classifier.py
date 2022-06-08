import numpy as np
import random

class Bayes:
    def __init__(self, data_num=10_000, data_dims=50):
        self.data_dims = data_dims
        self.data_num = data_num
        self.train_data = np.zeros(( data_num, data_dims ))
        self.train_label = np.zeros(( data_num, 1 ))
        self.test_data = np.zeros(( data_num, data_dims ))
        self.test_label = np.zeros(( data_num, 1 ))
        self.p_prior = []
        self.mean = []
        self.cov = []
        self.error_num = 0

    def get_dataset(self):
        ### get train_data, train_label, test_data, test_label ###

        # get data0(category0), data1(category1)
        data0 = np.load('data0.npy')
        data1 = np.load('data1.npy')

        size = data0.shape[0]

        # get label
        label0 = np.full(shape=(size,1), fill_value=0, dtype=np.int32)
        label1 = np.full(shape=(size,1), fill_value=1, dtype=np.int32)

        # merge data/label of category1 and category2
        data_all = np.concatenate((data0, data1))
        label_all = np.concatenate((label0, label1))

        # shuffle all data
        temp = list(zip(data_all, label_all))
        random.shuffle(temp)
        data_shuf_tuple, label_shuf_tuple = zip(*temp)
        data_shuf = np.asarray(data_shuf_tuple)
        label_shuf = np.asarray(label_shuf_tuple)

        # save data/label in class
        self.train_data = data_shuf[:size]
        self.train_label = label_shuf[:size]
        self.test_data = data_shuf[-size:]
        self.test_label = label_shuf[-size:]

        return
    
    def calcu_p_prior(self):
        ### calculate priori probability, mean, covariance of each category ###

        # separate categories
        temp = list(zip( self.train_data, self.train_label ))
        cate0_list = [ temp[i][0] for i in range(len(temp)) if temp[i][1].item()==0 ]
        cate1_list = [ temp[i][0] for i in range(len(temp)) if temp[i][1].item()==1 ]
        cate0 = np.array(cate0_list)
        cate1 = np.array(cate1_list)
        
        # calcu p_prior
        self.p_prior.append( cate0.shape[0] / self.data_num )
        self.p_prior.append( cate1.shape[0] / self.data_num )
    
        # calcu mean
        self.mean.append( np.mean( cate0, axis=0 ))
        self.mean.append( np.mean( cate1, axis=0 ))
        
        # calcu cov
        self.cov.append( np.cov( cate0.T ))
        self.cov.append( np.cov( cate1.T ))

    def calcu_p_normal(self, feature_x):
        det_cov0 = np.linalg.det(self.cov[0])
        det_cov1 = np.linalg.det(self.cov[1])

        inv_cov0 = np.linalg.inv(self.cov[0])
        inv_cov1 = np.linalg.inv(self.cov[1])

        d0 = ((feature_x - self.mean[0]).T).dot(inv_cov0).dot(feature_x - self.mean[0])
        d1 = ((feature_x - self.mean[1]).T).dot(inv_cov1).dot(feature_x - self.mean[1])

        p_normal0 = ( 1/ (((2*np.pi)**(self.data_dims/2)) * det_cov0**(1/2))) * np.exp((-1/2) * d0)
        p_normal1 = ( 1/ (((2*np.pi)**(self.data_dims/2)) * det_cov1**(1/2))) * np.exp((-1/2) * d1)

        return p_normal0, p_normal1

    def calcu_p_post(self, feature_x):
        ### not really posterior probability. It's p(x|w)*p(w) = p(w|x)*p(x), because p(x) is the same in w1 amd w2 ###
        
        p_normal0, p_normal1 = self.calcu_p_normal( feature_x )
        
        return (p_normal0 * self.p_prior[0], 
                p_normal1 * self.p_prior[1])

    def predict_error_rate(self):
        ### predict each data in test_data, and calculate error rate ###

        for i in range(self.test_data.shape[0]):
            
            feature_x = self.test_data[i]
            label = self.test_label[i].item()

            p_post0, p_post1 = self.calcu_p_post(feature_x)

            # if predict wrong then error +1
            if((p_post0 < p_post1) != label):
                self.error_num += 1
        
        print( 'error rate :', self.error_num / self.test_data.shape[0] )

bayes = Bayes()
bayes.get_dataset() # get train_data, train_label, test_data, test_label, and save them in class
bayes.calcu_p_prior() # calculate priori probability, mean, covariance of each category, and save them in class
bayes.predict_error_rate() # predict each data in test_data, and calculate error rate
