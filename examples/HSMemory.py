from HSElement import HSElement
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import series_to_supervised, split_data
from sklearn.metrics.regression import mean_squared_error
from TensorGlobal import TensorGlobal
import tensorflow as tf
import numpy as np


def get_worst_element():
    return HSMemory.hmMemory[HSMemory.max_index]


class HSMemory:
    number_decision_var = 4  # need review this
    hmMemory = []
    HMCR = [0.8, 0.8, 0.8, 0.8]  # for each element
    PAR = [0.4, 0.4, 0.4, 0.4]  # for each element
    max_mse = 0
    min_mse = 1000
    max_index = 0  # index for worst element
    min_index = 0  # index for best element
    last_index = 0
    HMS = 10
    better_flg = 0  # index for newest element
    best_flg = 0
    last_HMCR = [0, 0, 0, 0]  # property need to change
    last_PAR = [0, 0, 0, 0]

    def __init__(self, tmp_list):
        self.init_harmony_memory(tmp_list)

    @staticmethod
    def create_train_and_test_data(tmp_list, number_inputs):
        tmp_data = series_to_supervised(tmp_list, number_inputs)
        tmp_data_input = []
        x = ['t-'+str(i) for i in range(1, 11)]
        for i in range(1, 11):
            if number_inputs == i:
                tmp_data_input = tmp_data[x[i-1::-1]].values

        tmp_data_labels = tmp_data['t'].values.reshape(-1, 1)
        return split_data(tmp_data_input, tmp_data_labels)

    @staticmethod
    def init_harmony_memory(tmp_list):
        for i in range(0, HSMemory.HMS):
            print('Init HS with i: %f' % i)
            tmp_element = HSElement()
            tmp_element.index = i  # set index in HMS for current element
            tmp_element = HSMemory.update_mse(tmp_element, tmp_list)
            # Set min
            if (HSMemory.min_mse == 1000) or (HSMemory.min_mse > tmp_element.test_mse):
                HSMemory.min_mse = tmp_element.test_mse
                HSMemory.min_index = i
                print('update min in Init min_index: %f' % i)
            # Set max
            if (HSMemory.max_mse == 0) or (HSMemory.max_mse < tmp_element.test_mse):
                HSMemory.max_mse = tmp_element.test_mse
                HSMemory.max_index = i
                print('update max in Init max_index: %f' % i)
            HSMemory.hmMemory.append(tmp_element)
            print('Finished Init HS with i: %f' % i)

    @staticmethod
    def update_mse(tmp_input_element, tmp_list):
        data_train, label_train, data_test, label_test = \
            HSMemory.create_train_and_test_data(tmp_list, tmp_input_element.number_visible_input)
        tmp_regression = SupervisedDBNRegression(
            hidden_layers_structure=[tmp_input_element.number_visible_input,
                                     tmp_input_element.number_hidden_input],
            learning_rate_rbm=tmp_input_element.learning_rate_rbm,
            learning_rate=tmp_input_element.learning_rate,
            n_epochs_rbm=tmp_input_element.n_epochs_rbm,
            n_iter_backprop=tmp_input_element.n_iter_back_prop,
            contrastive_divergence_iter=tmp_input_element.contrastive_divergence_iter,
            batch_size=tmp_input_element.batch_size,
            activation_function=tmp_input_element.activation_function,
            n_hidden_layers_mlp=tmp_input_element.n_hidden_layers_mlp,
            cost_function_name=tmp_input_element.cost_function_name)

        tmp_regression.fit(data_train, label_train)  # train data
        tmp_input_element.train_mse = sum(tmp_regression.train_loss) / HSElement.config_n_iter_back_prop

        y_pred_test = tmp_regression.predict(data_test)
        check_nan = np.isnan(y_pred_test).any()

        if check_nan:
            tmp_input_element.test_mse = 1000
        else:
            tmp_input_element.test_mse = mean_squared_error(label_test, y_pred_test)
        if np.isnan(tmp_input_element.train_mse) or np.isinf(tmp_input_element.train_mse):
            tmp_input_element.train_mse = 1000

        # add to export result
        tmp_result_data = [tmp_input_element.learning_rate_rbm,
                           tmp_input_element.learning_rate, tmp_input_element.number_visible_input,
                           tmp_input_element.number_hidden_input, tmp_input_element.train_mse,
                           tmp_input_element.test_mse, '', '', '', '', '', '', '', '']
        TensorGlobal.followHs.append(tmp_result_data)

        TensorGlobal.sessFlg = True
        tf.reset_default_graph()
        del tmp_regression
        return tmp_input_element

    # getting the values
    @staticmethod
    def get_hs_memory():
        print('Getting value')
        return HSMemory.hmMemory

    @staticmethod
    def get_best_element():
        return HSMemory.hmMemory[HSMemory.min_index]

    @staticmethod
    def update_max_index():
        tmp_max_element = HSMemory.hmMemory[HSMemory.max_index]
        HSMemory.max_mse = tmp_max_element.test_mse

        for i in range(0, HSMemory.HMS):
            tmp_element = HSMemory.hmMemory[i]
            if HSMemory.max_mse < tmp_element.test_mse:
                HSMemory.max_mse = tmp_element.test_mse
                HSMemory.max_index = i
