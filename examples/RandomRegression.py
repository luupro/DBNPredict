from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegression:
    number_visible_input = randint(1, 10)
    number_hidden_input = randint(1, 10)
    #number_iter_backprop = 1500 #lorenz
    #number_iter_backprop = 1200
    number_iter_backprop = 500
    number_iter_rbm_loop = 20
    tmp_learning_rate_rbm = 0.045
    tmp_learning_rate = 0.001

    @staticmethod
    def create_random_model():
        #tmp_learning_rate_rbm = 0.001 + uniform(0, 1) * (0.5 - 0.001) #lorenz
        RandomRegression.tmp_learning_rate_rbm = 0.0001 + uniform(0, 1) * (0.95 - 0.0001) #random
        #tmp_learning_rate = 0.01 + uniform(0, 1) * (0.9 - 0.01) #lorenz
        RandomRegression.tmp_learning_rate = 0.0001 + uniform(0, 1) * (0.95 - 0.0001) #random
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[RandomRegression.number_visible_input,
                                                                         RandomRegression.number_hidden_input],
                                                learning_rate_rbm=RandomRegression.tmp_learning_rate_rbm,
                                                learning_rate=RandomRegression.tmp_learning_rate,
                                                #n_epochs_rbm=100, #lorenz
                                                #n_epochs_rbm=150,
                                                #n_epochs_rbm=30,
                                                n_epochs_rbm=RandomRegression.number_iter_rbm_loop,
                                                n_iter_backprop=RandomRegression.number_iter_backprop,
                                                #n_iter_backprop=50,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor, RandomRegression.tmp_learning_rate_rbm, RandomRegression.tmp_learning_rate
