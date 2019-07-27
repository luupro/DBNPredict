from random import *


class HSElement:
    #  standard values
    config_hls = 5
    config_lrr = 0.01
    config_lrr_range_err = 0.005
    config_lr = 0.03
    config_lr_range_err = 0.005
    config_ner = 100
    config_n_iter_back_prop = 1500
    config_cdi = 2
    config_bs = 32
    config_af = 'relu'
    config_nhlm = 1
    config_cfn = 'mse'
    learning_rate_rbm = 0
    learning_rate = 0

    def __init__(self):
        random_var = uniform(0, 1)
        self.hidden_layers_structure = [self.config_hls]
        self.learning_rate_rbm = (self.config_lrr - self.config_lrr_range_err) + \
                                 (random_var * 2 * self.config_lrr_range_err)
        self.learning_rate = (self.config_lr - self.config_lr_range_err) + \
                             (random_var * 2 * self.config_lr_range_err)
        self.n_epochs_rbm = self.config_ner
        self.n_iter_back_prop = self.config_n_iter_back_prop
        self.contrastive_divergence_iter = self.config_cdi
        self.batch_size = self.config_bs  # 32
        self.activation_function = 'relu'
        self.n_hidden_layers_mlp = self.config_nhlm
        self.cost_function_name = 'mse'
        self.train_lost = 1000
