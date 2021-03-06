import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class tune_param_regressor:
    def __init__(self, x, y, epoch,learning_rate, no_of_layers, 
                 batch_size,validation_data):
        
        self.x = x
        self.y = y
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.no_of_layers = no_of_layers
        self.batch_size = batch_size
        self.validation_data = validation_data
        
        self.x_dimension = x.shape
        self.target_dimension = y.shape
        
        #self layers range (start, stop, step)
        self.layers_range = None
        
    
        self.possibilities = 0
        #call evaluate classifier to evaluate the network

        """
        call self.evaluate_classifier() to evaluate the network
        """  
                
        return
    
    
    def evaluate_regressor(self):
        #no of possiblities
        self.possibilities = self.no_of_posibilities(len(self.layers_range))
        
        # evaluate network
        a = self.x
        b = self.y
        c = self.epoch
        d = self.no_of_layers
        e = self.learning_rate
        f = self.batch_size
        g = self.validation_data
        
        dict_scores = self.evaluate_network(a,b,c,d,e,f,g)
        
        #find param with minimum variance
        result = self.minimum_variance(dict_scores)
        return result
    
    
    def minimum_variance(self, dict_scores):
        total_var = []
        for i in dict_scores.keys():
            variances = [np.var([np.var(m),np.var(n)]) for m,n in zip(dict_scores[i]['training_values'],dict_scores[i]['testing_values'])]
            total_var.append(np.var(variances))
        result = {}
        for key, value in zip(dict_scores.keys(),total_var):  
            if value == min(total_var):
                result = dict_scores[key]
                break
         
        #no of layers
        print('no of layers: ', result['no_of_layers'])
        print('Neuron values: ', result['neuron_values'])
        print('Best Optimizer: ', result['optimizer'])
        print('Best Loss: ', result['loss'])
        #print dataframe
        dict_training = {'training_value':result['training_values'],
                      'testing_value':result['testing_values']}
        indexes = ['accuracy', 'precision', 'recall','f1_score']
        frame = pd.DataFrame(dict_training, index = indexes)
        print(frame.head())
        return 
    
    
    def no_of_posibilities(self, no_of_layers):
        count = 1
        for i in range(1, no_of_layers+1):
            count *= i
        return count
    
    
    def create_network(self, neurons):
        model = tf.keras.Sequential()
        
        # iterate through the dict layers
        model.add(layers.Dense(neurons[0],input_shape=(self.x_dimension[1],), activation='relu'))
        
        # add hidden layers
        if len(neurons[1:]) == 1:
            model.add(layers.Dense(neurons[1:], activation= 'relu'))
        else:
            for i in range(neurons[1:]):
                model.add(layers.Dense(i, activation='relu'))

        # add output layers
        model.add(layers.Dense(1, activation = None))
            
        return model
    
    """
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here
    Issue here with _compute

    """
   
    
    def _compute(self, actual, prediction):
        with tf.Session() as sess:
            accuracy = accuracy_score(actual, prediction)
            f1_score = f1_score(actual, prediction)
            precision = precision_score(actual, prediction)
            recall = (precision *f1_score)/((2*precision)-f1_score)
            accuracy = accuracy.eval()
            recall = recall.eval()
            precision = precision.eval()
            f1_score = f1_score.eval()
        return accuracy, f1_score, precision, recall
    
    
    
    
    
    def evaluate_network(self,x,y,epoch,no_of_layers,
                         learning_rate,batch_size, 
                         validation_data):
        # ranges of neuron values
        ba, ma, ka = self.layers_range
        data = [i for i in range(ba, ma, ka)]
        
        #optimizers
        optimize = {
            'rmsprop':tf.keras.optimizers.RMSprop(learning_rate),
            'gradient descent':tf.keras.optimizers.SGD(learning_rate),
            'adam':tf.keras.optimizers.Adam(learning_rate),
            'adagrad':tf.keras.optimizers.Adagrad(learning_rate),
            'adadelta':tf.keras.optimizers.Adadelta(learning_rate),
            'nadem':tf.keras.optimizers.Nadam(learning_rate)
        }
        
        
        # losses
        losses = {
            'mse':tf.keras.losses.MSE,
            'mae':tf.keras.losses.MAE,
            'mape':tf.keras.losses.MAPE,
        }
        
        

        dict_neurons = {}
        for pos in tqdm(range(self.possibilities)):
            time.sleep(0.1)
            #randomly initialize values as numbers of neurons
            current_neurons = np.random.choice(data, no_of_layers, replace = False)
            
            
            #create network for the neurons
            model = self.create_network(current_neurons)
            
            #Train the network on available optimizers and losses
            for m in optimize.keys():
                #document training params
                dict_doc = {}
                model.compile(optimizer=optimize[m],
                              loss= losses['mae'],
                              metrics=['accuracy'])
                model.fit(x,y, epochs = epoch, batch_size= batch_size,
                         validation_data = validation_data, verbose = 0)
                pred_train, pred_test = model.predict(x), model.predict(validation_data[0])

                train_result = self._compute(y, pred_train)
                test_result = self._compute(validation_data[1],pred_test)

                # document training params
                dict_doc['no_of_layers'] = no_of_layers
                dict_doc['neuron_values'] = current_neurons
                dict_doc['optimizer'] = m
                dict_doc['loss'] = 'mean squared error'
                dict_doc['training_values'] = train_result
                dict_doc['testing_values'] = test_result
                  
                # take record of all iterations
                dict_neurons[str(pos)+'_'+m] = dict_doc
                    
            
        return dict_neurons