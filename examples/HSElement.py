from random import *
from examples.HSMemory import HSMemory


class HSElement:
    def __init__(self, par_flg):
        # tmp_int = randint(4, 6)
        tmp_int = 5
        random_var = uniform(0, 1)
        self.hidden_layers_structure = [tmp_int]
        if not par_flg:
            self.learning_rate_rbm = 0.01 + random_var * (0.1 - 0.01)  # from 0.01 to 0.01(*)
            self.learning_rate = 0.03 + random_var * (0.1 - 0.03)  # from 0.03(*) to 0.1
            self.n_epochs_rbm = 50 + random_var * (100 - 50)  # from 50 to 100(*)
            self.n_epochs_rbm = int(self.n_epochs_rbm)
            self.n_iter_back_prop = 500 + random_var * (1500 - 500)  # from 500 to 1500(*)
            self.n_iter_back_prop = int(self.n_iter_back_prop)
            self.contrastive_divergence_iter = 1 + random_var * (2 - 1)  # from 1 to 2(*)
            self.contrastive_divergence_iter = int(self.contrastive_divergence_iter)
        else:
            # learning_rate_rbm_index
            learning_rate_rbm_index = random.randint(0, 49)
            tmp_hm_element = HSMemory.hmMemory[learning_rate_rbm_index]
            self.learning_rate_rbm = tmp_hm_element.learning_rate_rbm
            # learning_rate
            learning_rate_index = random.randint(0, 49)
            tmp_hm_element = HSMemory.hmMemory[learning_rate_index]
            self.learning_rate = tmp_hm_element.learning_rate
            # n_epochs_rbm
            n_epochs_rbm_index = random.randint(0, 49)
            tmp_hm_element = HSMemory.hmMemory[n_epochs_rbm_index]
            self.n_epochs_rbm = tmp_hm_element.n_epochs_rbm
            # n_iter_back_prop
            n_iter_back_prop_index = random.randint(0, 49)
            tmp_hm_element = HSMemory.hmMemory[n_iter_back_prop_index]
            self.n_iter_back_prop = tmp_hm_element.n_iter_back_prop
            # contrastive_divergence_iter
            contrastive_divergence_iter_index = random.randint(0, 49)
            tmp_hm_element = HSMemory.hmMemory[contrastive_divergence_iter_index]
            self.contrastive_divergence_iter = tmp_hm_element.contrastive_divergence_iter
        self.batch_size = 32  # 32
        self.activation_function = 'relu'
        self.n_hidden_layers_mlp = 0
        self.cost_function_name = 'mse'
        self.worst_flg = False
        self.best_flg = False
        self.index = 0
        self.train_lost = 100
