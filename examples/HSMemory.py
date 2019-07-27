from examples.HSElement import HSElement
from dbn.tensorflow import SupervisedDBNRegression


def get_worst_element():
    return HSMemory.hmMemory[HSMemory.max_index]


class HSMemory:
    number_decision_var = 2  # need review this
    hmMemory = []
    HMCR = 0.3
    PAR = 0.3
    max_train_lost = 0
    min_train_lost = 1000
    max_index = 0  # index for worst element
    min_index = 0  # index for best element
    HMS = 7

    def __init__(self, data_train, label_train):
        self.init_harmony_memory(data_train, label_train)

    @staticmethod
    def init_harmony_memory(data_train, label_train):
        for i in range(0, HSMemory.HMS):
            print('Init HS with i: %f' % i)
            tmp_element = HSElement()
            tmp_element.index = i  # set index in HMS for current element
            tmp_element = HSMemory.update_train_lost(tmp_element, data_train, label_train)
            # Set min
            if (HSMemory.min_train_lost == 1000) or (HSMemory.min_train_lost > tmp_element.train_lost):
                HSMemory.min_train_lost = tmp_element.train_lost
                HSMemory.min_index = i
                print('update min in Init min_index: %f' % i)
            # Set max
            if (HSMemory.max_train_lost == 0) or (HSMemory.max_train_lost < tmp_element.train_lost):
                HSMemory.max_train_lost = tmp_element.train_lost
                HSMemory.max_index = i
                print('update max in Init max_index: %f' % i)
            HSMemory.hmMemory.append(tmp_element)
            print('Finished Init HS with i: %f' % i)

    @staticmethod
    def update_train_lost(tmp_input_element, data_train, label_train):
        tmp_return_element = SupervisedDBNRegression(hidden_layers_structure=tmp_input_element.hidden_layers_structure,
                                                 learning_rate_rbm=tmp_input_element.learning_rate_rbm,
                                                 learning_rate=tmp_input_element.learning_rate,
                                                 n_epochs_rbm=tmp_input_element.n_epochs_rbm,
                                                 n_iter_backprop=tmp_input_element.n_iter_back_prop,
                                                 contrastive_divergence_iter=tmp_input_element.contrastive_divergence_iter,
                                                 batch_size=tmp_input_element.batch_size,
                                                 activation_function=tmp_input_element.activation_function,
                                                 n_hidden_layers_mlp=tmp_input_element.n_hidden_layers_mlp,
                                                 cost_function_name=tmp_input_element.cost_function_name)
        tmp_return_element.fit(data_train, label_train)  # train data
        tmp_input_element.train_lost = sum(tmp_return_element.train_loss) / HSElement.config_n_iter_back_prop
        del tmp_return_element
        return tmp_input_element

    @staticmethod
    def train_data_and_return_model(tmp_input_element, data_train, label_train):
        tmp_return_element = SupervisedDBNRegression(hidden_layers_structure=tmp_input_element.hidden_layers_structure,
                                                     learning_rate_rbm=tmp_input_element.learning_rate_rbm,
                                                     learning_rate=tmp_input_element.learning_rate,
                                                     n_epochs_rbm=tmp_input_element.n_epochs_rbm,
                                                     n_iter_backprop=tmp_input_element.n_iter_back_prop,
                                                     contrastive_divergence_iter=tmp_input_element.contrastive_divergence_iter,
                                                     batch_size=tmp_input_element.batch_size,
                                                     activation_function=tmp_input_element.activation_function,
                                                     n_hidden_layers_mlp=tmp_input_element.n_hidden_layers_mlp,
                                                     cost_function_name=tmp_input_element.cost_function_name)
        tmp_return_element.fit(data_train, label_train)  # train data
        return tmp_return_element

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
        HSMemory.max_train_lost = tmp_max_element.train_lost

        for i in range(0, HSMemory.HMS):
            tmp_element = HSMemory.hmMemory[i]
            if HSMemory.max_train_lost < tmp_element.train_lost:
                HSMemory.max_train_lost = tmp_element.train_lost
                HSMemory.max_index = i
