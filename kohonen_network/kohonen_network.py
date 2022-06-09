import numpy as np
import random

class Data:
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    def get_dataset(self, filename0, filename1):
        ### get train_data, train_label, test_data, test_label ###

        # get data0(category0), data1(category1)
        data0 = np.load( filename0 )
        data1 = np.load( filename1 )

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
    
    def draw( self, draw_name, data, label ): ###############
        pass


class Kohonen:
    ### a kohonen network with only two layers, one is the 50 dims input, the other is the 2 dims output ###
    def __init__(self, input_shape, output_dims, learn_rate=0.5, max_iter_data_times=1000):
        input_num, input_dims = input_shape

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weights = np.random.sample(( input_dims, output_dims ))
        weights = self.weights / np.linalg.norm(self.weights)
        self.weights = weights

        self.learn_rate = learn_rate
        self.max_iteration = input_num * max_iter_data_times

    def get_winner( self, input_x ):
        neuron = np.zeros(self.output_dims)
        for ni in range( neuron.shape[0] ):
            for xi in range( input_x.shape[0] ):
                neuron[ni] += self.weights[xi,ni] * input_x[xi]
        winner = np.argmax(neuron)
        return winner

    def update_weight( self, winner, input_x, attempts_thres=1e-5 ):
        new_weights = self.weights[:,winner] + self.learn_rate*( input_x - self.weights[:,winner] )
        if ( np.linalg.norm(new_weights) < attempts_thres ):
            return False
        else:
            print(np.linalg.norm(new_weights))
            new_weights_norm = new_weights / np.linalg.norm(new_weights)
            self.weights[:,winner] = new_weights_norm
            return True

    def fit( self, data ):
        for i in range( self.max_iteration ):
            print(i, end=' ')
            input_x = data[i%data.shape[0]] / np.linalg.norm(data[i%data.shape[0]])
            winner = self.get_winner( input_x )
            if_update = self.update_weight( winner, input_x )
            #self.learn_rate -= i/self.max_iteration
            if(not if_update):
                print('No significant change so stop fitting\nAt iteration ', i, '/', self.max_iteration, sep='')
                break
        print('End of all iteration.')

    def predict( self, data ): ###########
        pass



data = Data()
data.get_dataset( 'data0.npy', 'data1.npy' )

input_shape = data.train_data.shape
output_dims = len(set(data.train_label.flatten()))
kohonen = Kohonen( input_shape, output_dims)

kohonen.fit( data.train_data )
print(kohonen.weights)
predict_label = kohonen.predict( data.test_data )
'''
data.draw( 'test_groundtruth.png', data.test_data, data.test_label )
data.draw( 'test_predict.png', data.test_data, predict_label )
'''