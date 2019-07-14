from examples.HSElement import HSElement
from dbn.tensorflow import SupervisedDBNRegression


class HSMemory:
    number_decision_var = 5

    def __init__(self, data_train, label_train):
        self.hmMemory = []
        self.max_index = 0  # index for worst element
        self.min_index = 0  # index for best element
        self.max_train_lost = 1000
        self.min_train_lost = 0
        self.HMS = 5
        self.HMCR = 0.7
        self.PAR = 0.7
        self.init_harmony_memory(data_train, label_train)

    def init_harmony_memory(self, data_train, label_train):
        for i in range(1, self.HMS):
            tmp_element = HSElement()
            tmp_element.index = i-1  # set index in HMS for current element
            tmp_element = HSMemory.train_with_element_option(tmp_element, data_train, label_train)
            # Set min
            if (self.min_train_lost == 1000) or (self.min_train_lost > tmp_element.train_lost):
                self.min_train_lost = tmp_element.train_lost
                self.min_index = i-1
            # Set max
            if (self.max_train_lost == 0) or (self.max_train_lost < tmp_element.train_lost):
                self.max_train_lost = tmp_element.train_lost
                self.max_index = i-1
            self.hmMemory.append(tmp_element)

    @staticmethod
    def train_with_element_option(tmp_input_element, data_train, label_train):
        tmp_return_element = SupervisedDBNRegression(hidden_layers_structure=tmp_input_element.hidden_layers_structure,
                                                 learning_rate_rbm=tmp_input_element.learning_rate_rbm,
                                                 learning_rate=tmp_input_element.learning_rate,
                                                 n_epochs_rbm=tmp_input_element.n_epochs_rbm,
                                                 n_iter_backprop=tmp_input_element.n_iter_backprop,
                                                 contrastive_divergence_iter=tmp_input_element.contrastive_divergence_iter,
                                                 batch_size=tmp_input_element.batch_size,
                                                 activation_function=tmp_input_element.activation_function,
                                                 n_hidden_layers_mlp=tmp_input_element.n_hidden_layers_mlp,
                                                 cost_function_name=tmp_input_element.cost_function_name)
        tmp_return_element.fit(data_train, label_train)  # train data
        tmp_input_element.train_lost = tmp_return_element.train_loss[-1]
        return tmp_input_element

    # getting the values
    def get_hs_memory(self):
        print('Getting value')
        return self.hmMemory

    def get_best_element(self):
        return self.hmMemory[self.min_index]

    def get_worst_element(self):
        return self.hmMemory[self.max_index]

