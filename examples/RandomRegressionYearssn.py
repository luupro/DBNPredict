from dbn.tensorflow import SupervisedDBNRegression
from random import *


class RandomRegressionYearssn:
    number_visible_input = 5 #randint(1, 10)
    number_hidden_input = 5 #randint(1, 10)
    number_iter_backprop = 500
    number_iter_rbm_loop = 10
    tmp_learning_rate_rbm = 0.05
    tmp_learning_rate = 0.008

    @staticmethod
    def create_random_model():
        RandomRegressionYearssn.number_visible_input = randint(1, 10)
        RandomRegressionYearssn.number_hidden_input = randint(1, 10)
        RandomRegressionYearssn.tmp_learning_rate_rbm = 0.001 + uniform(0, 1) * (0.1 - 0.001) #random
        RandomRegressionYearssn.tmp_learning_rate = 0.003 + uniform(0, 1) * (0.3 - 0.003) #random
        tmp_regressor = SupervisedDBNRegression(hidden_layers_structure=[RandomRegressionYearssn.number_visible_input,
                                                                         RandomRegressionYearssn.number_hidden_input],
                                                learning_rate_rbm=RandomRegressionYearssn.tmp_learning_rate_rbm,
                                                learning_rate=RandomRegressionYearssn.tmp_learning_rate,
                                                n_epochs_rbm=RandomRegressionYearssn.number_iter_rbm_loop,
                                                n_iter_backprop=RandomRegressionYearssn.number_iter_backprop,
                                                contrastive_divergence_iter=2,
                                                batch_size=32,
                                                activation_function='relu',
                                                n_hidden_layers_mlp=1,
                                                cost_function_name='mse')
        return tmp_regressor, RandomRegressionYearssn.tmp_learning_rate_rbm, RandomRegressionYearssn.tmp_learning_rate
