from datetime import datetime
from random import *

import numpy as np
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler

from dbn.utils import read_file
from examples.HSElement import HSElement
from examples.HSMemory import HSMemory
from examples.TensorGlobal import TensorGlobal


def transfer_hs_memory_data(tmp_hs_memory):  # print HS
    tmp_all_data = []
    for ii in range(0, len(tmp_hs_memory)):
        tmp1_hs_element = tmp_hs_memory[ii]
        tmp_lrr = tmp1_hs_element.learning_rate_rbm
        tmp_lr = tmp1_hs_element.learning_rate
        tmp_number_visible = tmp1_hs_element.number_visible_input
        tmp_number_hidden = tmp1_hs_element.number_hidden_input
        tmp_train_mse = tmp1_hs_element.train_mse
        tmp_test_mse = tmp1_hs_element.test_mse
        tmp1_element_data = [tmp_lrr, tmp_lr, tmp_number_visible, tmp_number_hidden,
                             tmp_train_mse, tmp_test_mse, '', '', '']
        tmp_all_data.append(tmp1_element_data)
    return tmp_all_data
# --------------------------------------------


path = 'chaotic-timeseries/Lorenz.txt'
xs = np.array(read_file(path))

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))
cost_function_name = 'mse'
# cost_function_name = 'mae'
# cost_function_name = 'mape'

# Loop for
# element = HSElement()
hs_memory_object = HSMemory(lorenz_scale.tolist())
# main_regression = SupervisedDBNRegression()
NI = 50  # number of improvisations


# Improve a new hamony
for h in range(0, NI):
    new_harmony_element = HSElement()  # already random for all variable
    for i in range(1, HSMemory.number_decision_var + 1):
        if uniform(0, 1) <= HSMemory.HMCR:
            if uniform(0, 1) > HSMemory.PAR:
                random_index = randint(0, HSMemory.HMS-1)
                tmp_hm_element = HSMemory.hmMemory[random_index]
            else:
                tmp_hm_element = HSMemory.hmMemory[HSMemory.min_index]
            if i == 1:
                new_harmony_element.learning_rate_rbm = HSElement.get_new_lrr(tmp_hm_element.learning_rate_rbm)
            if i == 2:
                new_harmony_element.learning_rate = HSElement.get_new_lr(tmp_hm_element.learning_rate)
            if i == 3:
                new_harmony_element.number_hidden_input = \
                    HSElement.get_new_number_input(tmp_hm_element.number_visible_input)
            if i == 4:
                new_harmony_element.number_hidden_input \
                    = HSElement.get_new_number_input(tmp_hm_element.number_hidden_input)

    # Step4 Update harmony memory
    print('Update train mse  for Improve harmony h: %f' % h)
    new_harmony_element = HSMemory. \
        update_mse(new_harmony_element, lorenz_scale.tolist())
    # compare with worst element
    if new_harmony_element.test_mse < HSMemory.max_mse:
        tmp_label_replace = 'replace worst - index_hs: ' + str(HSMemory.max_index)
        tmp_label_new_min = ''
        HSMemory.hmMemory[HSMemory.max_index] = new_harmony_element  # replace worst
        if HSMemory.min_mse > new_harmony_element.test_mse:
            HSMemory.min_mse = new_harmony_element.test_mse  # update min mse
            HSMemory.min_index = HSMemory.max_index  # update min_index
            tmp_label_new_min = ' - new_min_index: ' + str(HSMemory.min_index)
        HSMemory.update_max_index()  # update max_index
        tmp_label_new_max = ' - new_max_index: ' + str(HSMemory.max_index)
        tmp_last_result = TensorGlobal.followHs[-1]
        tmp_last_result[-1] = tmp_label_new_min
        tmp_last_result[-2] = tmp_label_new_max
        tmp_last_result[-3] = tmp_label_replace
        # print('NEw max_index: %f' % HSMemory.max_index)

result_data = transfer_hs_memory_data(HSMemory.hmMemory)


def export_result(file_name, tmp_array):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    workbook = xlsxwriter.Workbook(file_name + dt_string + '.xlsx')
    worksheet = workbook.add_worksheet()
    row = 1
    col = 0

    for learning_rate_rbm, learning_rate, number_visible_input, number_visible_hidden, \
        mse_train, mse_test, label_replace, label_max, label_min \
            in tmp_array:
        worksheet.write(row, col, learning_rate_rbm)
        worksheet.write(row, col + 1, learning_rate)
        worksheet.write(row, col + 2, number_visible_input)
        worksheet.write(row, col + 3, number_visible_hidden)
        worksheet.write(row, col + 4, mse_train)
        worksheet.write(row, col + 5, mse_test)
        worksheet.write(row, col + 6, label_replace)
        worksheet.write(row, col + 7, label_max)
        worksheet.write(row, col + 8, label_min)
        row += 1
    workbook.close()


# export_result('result_hs_memory', result_data)
export_result('result_hs_full_', TensorGlobal.followHs)
