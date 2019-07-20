from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegression:
    @staticmethod
    def create_random_model():
        random_var = uniform(0, 1)
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[5],
                                                learning_rate_rbm=0.005 + random_var * (0.015 - 0.005),
                                                learning_rate=0.02 + random_var * (0.04 - 0.02),
                                                n_epochs_rbm=100,
                                                n_iter_backprop=1500,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor
