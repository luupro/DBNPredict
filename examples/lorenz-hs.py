import numpy as np
from examples.HSMemory import HSMemory
from examples.HSElement import HSElement
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import read_file
from random import *


def transfer_hs_memory_data(tmp_hs_memory):  # print HS
    tmp_all_data = []
    tmp_header_data = ('learning_rate_rbm', 'learning_rate', 'mse_train', 'mse_test')
    for ii in range(0, len(tmp_hs_memory)):
        tmp1_hs_element = tmp_hs_memory[ii]
        tmp_lrr = tmp1_hs_element.learning_rate_rbm
        tmp_lr = tmp1_hs_element.learning_rate
        tmp_mse = tmp1_hs_element.mse
        tmp1_element_data = [tmp_lrr, tmp_lr, tmp_mse, 'No Count']
        tmp_all_data.append(tmp1_element_data)
    return tmp_all_data, tmp_header_data
# --------------------------------------------


path = 'chaotic-timeseries/Lorenz3.txt'
xs = np.array(read_file(path))

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))
cost_function_name = 'mse'
# cost_function_name = 'mae'
# cost_function_name = 'mape'

# Loop for
element = HSElement()
hs_memory_object = HSMemory(lorenz_scale.tolist())
main_regression = SupervisedDBNRegression()
NI = 4  # number of improvisations


# Improve a new hamony
new_harmony_element = HSElement()  # already random for all variable

result_data, result_label = transfer_hs_memory_data(HSMemory.hmMemory)

for h in range(0, NI):
    for i in range(1, HSMemory.number_decision_var + 1):
        if uniform(0, 1) <= HSMemory.HMCR:
            if uniform(0, 1) > HSMemory.PAR:
                random_index = randint(0, HSMemory.HMS-1)
                tmp_hm_element = HSMemory.hmMemory[random_index]
            else:
                tmp_hm_element = HSMemory.hmMemory[HSMemory.min_index]
            if i == 1:
                new_harmony_element.learning_rate_rbm = \
                    tmp_hm_element.learning_rate_rbm + uniform(-HSElement.config_lrr_range_err,
                                                               HSElement.config_lrr_range_err)
            if i == 2:
                new_harmony_element.learning_rate = \
                    tmp_hm_element.learning_rate + uniform(-HSElement.config_lr_range_err,
                                                           HSElement.config_lr_range_err)
            if i == 3:
                new_harmony_element.config_number_hidden_input = tmp_hm_element.config_number_visible_input
            if i == 4:
                new_harmony_element.config_number_hidden_input = tmp_hm_element.config_number_hidden_input

    # Step4 Update harmony memory
    print('Update train mse  for Improve harmony h: %f' % h)
    new_harmony_element = HSMemory. \
        update_mse(new_harmony_element, lorenz_scale.tolist())
    # compare with worst element
    if new_harmony_element.mse < HSMemory.max_mse:
        HSMemory.hmMemory[HSMemory.max_index] = new_harmony_element  # replace worst
        print('replace max_index: %f' % HSMemory.max_index)
        print('new mse: %f' % new_harmony_element.mse)
        if HSMemory.min_mse > new_harmony_element.mse:
            HSMemory.min_mse = new_harmony_element.mse  # update min mse
            HSMemory.min_index = HSMemory.max_index  # update min_index
            print('NEw min_index: %f' % HSMemory.min_index)
        HSMemory.update_max_index()  # update max_index
        print('NEw max_index: %f' % HSMemory.max_index)

result_data, result_label = transfer_hs_memory_data(HSMemory.hmMemory)
# get best Harmory
tmp_hs_element = HSMemory.hmMemory[HSMemory.min_index]
tmp_element_data = [tmp_hs_element.learning_rate_rbm,
                    tmp_hs_element.learning_rate, tmp_hs_element.mse]
result_data.append(tmp_element_data)
