from random import *


class HSElement:
    #  standard values
    config_hls = 5
    config_lrr = 0.01
    config_lr = 0.03
    config_ner = 100
    config_nib = 1500
    config_cdi = 2
    config_bs = 32
    config_af = 'relu'
    config_nhlm = 1
    config_cfn = 'mse'

    def __init__(self):
        random_var = uniform(0, 1)
        self.hidden_layers_structure = [self.config_hls]
        self.learning_rate_rbm = self.config_lrr + random_var * (0.1 - 0.01)  # from 0.01 to 0.01(*)
        self.learning_rate = self.config_lr + random_var * (0.1 - 0.03)  # from 0.03(*) to 0.1
        self.n_epochs_rbm = self.config_ner
        self.n_iter_back_prop = self.config_nib
        self.contrastive_divergence_iter = self.config_cdi
        self.batch_size = self.config_bs  # 32
        self.activation_function = 'relu'
        self.n_hidden_layers_mlp = self.config_nhlm
        self.cost_function_name = 'mse'
        self.train_lost = 1000
