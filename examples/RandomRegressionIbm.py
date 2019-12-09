from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegressionIbm:
    number_visible_input = randint(1, 10)
    number_hidden_input = randint(1, 10)
    number_iter_backprop = 4000
    number_iter_rbm_loop = 50
    tmp_learning_rate_rbm = 0.01
    tmp_learning_rate = 0.0005

    @staticmethod
    def create_random_model():
        RandomRegressionIbm.tmp_learning_rate_rbm = 0.001 + uniform(0, 1) * (0.1 - 0.001) #random
        RandomRegressionIbm.tmp_learning_rate = 0.0005 + uniform(0, 1) * (0.05 - 0.0005) #random
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[RandomRegressionIbm.number_visible_input,
                                                                         RandomRegressionIbm.number_hidden_input],
                                                learning_rate_rbm=RandomRegressionIbm.tmp_learning_rate_rbm,
                                                learning_rate=RandomRegressionIbm.tmp_learning_rate,
                                                n_epochs_rbm=RandomRegressionIbm.number_iter_rbm_loop,
                                                n_iter_backprop=RandomRegressionIbm.number_iter_backprop,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor, RandomRegressionIbm.tmp_learning_rate_rbm, RandomRegressionIbm.tmp_learning_rate
