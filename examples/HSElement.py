from random import *
from examples.TensorGlobal import TensorGlobal

class HSElement:
    #  standard values
    config_lrr_lb = 0.001
    config_lrr_ub = 0.1
    config_lrr_range_min = 0.00001
    config_lrr_range_max = 0.0001
    config_lr_lb = 0.005
    config_lr_ub = 0.05
    config_lr_range_min = 0.00001
    config_lr_range_max = 0.0001
    #config_ner = 100 #lorenz
    #config_ner = 150
    config_ner = 15
    #config_n_iter_back_prop = 1500 #lorenz
    #config_n_iter_back_prop = 1200
    config_n_iter_back_prop = 100
    config_cdi = 2
    config_bs = 32
    config_af = 'relu'
    config_nhlm = 1
    config_cfn = 'mse'

    def __init__(self):
        self.number_visible_input = randint(1, 10)
        self.number_hidden_input = randint(1, 10)
        
        # retrieve through value in the range
        if uniform(0, 1) < 0.9:
            self.learning_rate_rbm = self.config_lrr_lb + uniform(0, 1) * (self.config_lrr_ub - self.config_lrr_lb)
        else:
            self.learning_rate_rbm = self.config_lrr_lb + (self.config_lrr_ub/100)*TensorGlobal.global_range_lr_rmb
            + randint(-1, 1) * self.config_lrr_range_min
            if TensorGlobal.global_range_lr_rmb == 100:
                TensorGlobal.global_range_lr_rmb = 0
            else:
                TensorGlobal.global_range_lr_rmb += 1

        if uniform(0, 1) < 0.5:
            self.learning_rate = self.config_lr_lb + uniform(0, 1) * (self.config_lr_ub - self.config_lr_lb)
        else:
            self.learning_rate = self.config_lr_lb + (self.config_lr_ub/100)*TensorGlobal.global_range_lr
            + randint(-1, 1) * self.config_lr_range_min
            if TensorGlobal.global_range_lr == 100:
                TensorGlobal.global_range_lr = 0
            else:
                TensorGlobal.global_range_lr += 1

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
