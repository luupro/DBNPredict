from random import *
from TensorGlobal import TensorGlobal

class HSElement:
    #  standard values
    config_lrr_lb = 0.001
    config_lrr_ub = 0.1
    config_lrr_range_min = 0.0001
    config_lrr_range_max = 0.001
    config_lr_lb = 0.003
    config_lr_ub = 0.03
    config_lr_range_min = 0.0001
    config_lr_range_max = 0.001
    config_ner = 10
    config_n_iter_back_prop = 200 #pci
    config_cdi = 2
    config_bs = 32
    config_af = 'relu'
    config_nhlm = 1
    config_cfn = 'mse'

    def __init__(self):
        self.number_visible_input = randint(1, 10)
        self.number_hidden_input = randint(1, 10)
        # retrieve through value in the range
        self.learning_rate_rbm = self.config_lrr_lb + uniform(0, 1) * (self.config_lrr_ub - self.config_lrr_lb)
        self.learning_rate = self.config_lr_lb + uniform(0, 1) * (self.config_lr_ub - self.config_lr_lb)
        self.n_epochs_rbm = self.config_ner
        self.n_iter_back_prop = self.config_n_iter_back_prop
        self.contrastive_divergence_iter = self.config_cdi
        self.batch_size = self.config_bs  # 32
        self.activation_function = 'relu'
        self.n_hidden_layers_mlp = self.config_nhlm
        self.cost_function_name = 'mse'
        self.train_mse = 1000
        self.test_mse = 1000

    # new improve lrr values
    @staticmethod
    def get_new_lrr(old_lrr):
        random_list = [-1, 1]
        while True:
            new_value = old_lrr + choice(random_list) \
                        * uniform(HSElement.config_lrr_range_min, HSElement.config_lrr_range_max)
            if HSElement.config_lrr_lb < new_value < HSElement.config_lrr_ub:
                break
        return new_value

    # new improve lr values
    @staticmethod
    def get_new_lr(old_lr):
        random_list = [-1, 1]
        while True:
            new_value = old_lr + choice(random_list) \
                        * uniform(HSElement.config_lr_range_min, HSElement.config_lr_range_max)
            if HSElement.config_lr_lb < new_value < HSElement.config_lr_ub:
                break
        return new_value

    # new improve lr values
    @staticmethod
    def get_new_number_input(old_lr):
        while True:
            new_value = old_lr + randint(-1, 1)
            if 0 < new_value < 11:
                break
        return new_value
