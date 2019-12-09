from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegressionDolar:
    number_visible_input = randint(1, 10)
    number_hidden_input = randint(1, 10)
    number_iter_backprop = 4000
    number_iter_rbm_loop = 50
    tmp_learning_rate_rbm = 0.005
    tmp_learning_rate = 0.0005

    @staticmethod
    def create_random_model():
        RandomRegressionDolar.tmp_learning_rate_rbm = 0.0005 + uniform(0, 1) * (0.05 - 0.0005) #random
        RandomRegressionDolar.tmp_learning_rate = 0.00005 + uniform(0, 1) * (0.005 - 0.00005) #random
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[RandomRegressionDolar.number_visible_input,
                                                                         RandomRegressionDolar.number_hidden_input],
                                                learning_rate_rbm=RandomRegressionDolar.tmp_learning_rate_rbm,
                                                learning_rate=RandomRegressionDolar.tmp_learning_rate,
                                                n_epochs_rbm=RandomRegressionDolar.number_iter_rbm_loop,
                                                n_iter_backprop=RandomRegressionDolar.number_iter_backprop,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor, RandomRegressionDolar.tmp_learning_rate_rbm, RandomRegressionDolar.tmp_learning_rate
