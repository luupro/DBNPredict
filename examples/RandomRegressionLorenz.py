from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegressionLorenz:
    number_visible_input = randint(1, 10)
    number_hidden_input = randint(1, 10)
    number_iter_backprop = 2000
    number_iter_rbm_loop = 20
    tmp_learning_rate_rbm = 0.01
    tmp_learning_rate = 0.03

    @staticmethod
    def create_random_model():
        RandomRegressionLorenz.tmp_learning_rate_rbm = 0.001 + uniform(0, 1) * (0.1 - 0.001) #random
        RandomRegressionLorenz.tmp_learning_rate = 0.003 + uniform(0, 1) * (0.3 - 0.003) #random
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[RandomRegressionLorenz.number_visible_input,
                                                                         RandomRegressionLorenz.number_hidden_input],
                                                learning_rate_rbm=RandomRegressionLorenz.tmp_learning_rate_rbm,
                                                learning_rate=RandomRegressionLorenz.tmp_learning_rate,
                                                n_epochs_rbm=RandomRegressionLorenz.number_iter_rbm_loop,
                                                n_iter_backprop=RandomRegressionLorenz.number_iter_backprop,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor, RandomRegressionLorenz.tmp_learning_rate_rbm, RandomRegressionLorenz.tmp_learning_rate
