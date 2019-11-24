from random import uniform
from random import randint

class Position:
    #  standard values
    config_lrr_max = 0.95
    config_lrr_min = 0.0001
    config_lr_max = 0.95
    config_lr_min = 0.0001
    config_ner = 20 #pci
    config_n_iter_back_prop = 500  #pci
    config_cdi = 2
    config_bs = 32
    config_af = 'relu'
    config_nhlm = 1
    config_cfn = 'mse'
    config_number_visible_input_max = 10
    config_number_visible_input_min = 1
    config_number_hidden_input_max = 10
    config_number_hidden_input_min = 1

    def __init__(self, num_particle, index):
        self.number_visible_input = randint(1, 10)
        self.number_hidden_input = randint(1, 10)
        if num_particle == 0:
            self.learning_rate_rbm = self.config_lrr_min + uniform(0, 1) * (self.config_lrr_max - self.config_lrr_min)
            self.learning_rate = self.config_lr_min + uniform(0, 1) * (self.config_lr_max - self.config_lr_min)
        else:
            self.learning_rate_rbm = self.config_lrr_max / num_particle * index + self.config_lrr_min
            self.learning_rate = self.config_lr_max / num_particle * index + self.config_lr_min
        self.n_epochs_rbm = self.config_ner
        self.n_iter_back_prop = self.config_n_iter_back_prop
        self.contrastive_divergence_iter = self.config_cdi
        self.batch_size = self.config_bs  # 32
        self.activation_function = 'relu'
        self.n_hidden_layers_mlp = self.config_nhlm
        self.cost_function_name = 'mse'
        self.train_mse = 1000
        self.test_mse = 1000
