from datetime import datetime
from random import *

import numpy as np
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler

from dbn.utils import read_file
from HSElement import HSElement
from HSMemory import HSMemory
from TensorGlobal import TensorGlobal


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

path = 'chaotic-timeseries/yearssn.txt'  # vdmt

xs = np.array(read_file(path))

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))
cost_function_name = 'mse'

# Loop for
hs_memory_object = HSMemory(lorenz_scale.tolist())
NI = 100  # number of improvisations

# Improve a new harmony
for h in range(0, NI):
    new_harmony_element = HSElement()  # already random for all variable
    min_hm_element = HSMemory.hmMemory[HSMemory.min_index]  # min HSM element
    # Change just one decision var in one time -- no use
    #for i in range(1, HSMemory.number_decision_var + 1):
    i = randint(1, HSMemory.number_decision_var)
        #HSMemory.last_HMCR[i-1] = 0
        #HSMemory.last_PAR[i-1] = 0
    if uniform(0, 1) <= HSMemory.HMCR:
        pit_index = randint(0, HSMemory.HMS-1)
        tmp_hm_element = HSMemory.hmMemory[pit_index]
        if i == 1:
            new_harmony_element.learning_rate_rbm = tmp_hm_element.learning_rate_rbm
        if i == 2:
            new_harmony_element.learning_rate = tmp_hm_element.learning_rate
        if i == 3:
            new_harmony_element.number_visible_input = tmp_hm_element.number_visible_input
        if i == 4:
            new_harmony_element.number_hidden_input = tmp_hm_element.number_hidden_input
        if uniform(0, 1) <= HSMemory.PAR:
            best_element = HSMemory.hmMemory[HSMemory.min_index]
            if i == 1:
                new_harmony_element.learning_rate_rbm = min_hm_element.learning_rate_rbm
            if i == 2:
                new_harmony_element.learning_rate = min_hm_element.learning_rate
            if i == 3:
                new_harmony_element.number_visible_input = min_hm_element.number_visible_input
            if i == 4:
                new_harmony_element.number_hidden_input = min_hm_element.number_hidden_input

    # Step4 Update harmony memory
    print('Update train mse  for Improve harmony h: %f' % h)
    new_harmony_element = HSMemory. \
        update_mse(new_harmony_element, lorenz_scale.tolist())
    # compare with worst element
    tmp_last_result = TensorGlobal.followHs[-1]
    if new_harmony_element.test_mse < HSMemory.max_mse:
        # keep parameter
        HSMemory.better_flg = 1
        HSMemory.last_index = HSMemory.max_index
        tmp_label_replace = 'replace worst - index_hs: ' + str(HSMemory.max_index)
        tmp_label_new_min = ''
        HSMemory.hmMemory[HSMemory.max_index] = new_harmony_element  # replace worst
        if HSMemory.min_mse > new_harmony_element.test_mse:
            HSMemory.best_flg = 1
            HSMemory.min_mse = new_harmony_element.test_mse  # update min mse
            HSMemory.min_index = HSMemory.max_index  # update min_index
            tmp_label_new_min = ' - new_min_index: ' + str(HSMemory.min_index)
        else:
            HSMemory.best_flg = 0
        HSMemory.update_max_index()  # update max_index
        tmp_label_new_max = ' - new_max_index: ' + str(HSMemory.max_index)
        tmp_last_result[-1] = tmp_label_new_min
        tmp_last_result[-2] = tmp_label_new_max
        tmp_last_result[-3] = tmp_label_replace
        # keep last index
        # print('NEw max_index: %f' % HSMemory.max_index)

    # for debug only
    for i in range(1, HSMemory.number_decision_var + 1):
        ii = i-8
        tmp_last_result[ii] = str(HSMemory.last_HMCR[i-1]) + "_" + str(HSMemory.last_PAR[i-1])
    tmp_last_result[-8] = HSMemory.last_index

result_data = transfer_hs_memory_data(HSMemory.hmMemory)

def export_result(file_name, tmp_array):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    workbook = xlsxwriter.Workbook(file_name + dt_string + '.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    # make header
    arr_header = []
    arr_header.append('learning_rate_rbm')
    arr_header.append('learning_rate')
    arr_header.append('number_visible_input')
    arr_header.append('number_visible_hidden')
    arr_header.append('mse_train')
    arr_header.append('mse_test')
    arr_header.append('last_index')
    arr_header.append('hmcr_par1')
    arr_header.append('hmcr_par2')
    arr_header.append('hmcr_par3')
    arr_header.append('hmcr_par4')
    arr_header.append('label_replace')
    arr_header.append('label_max')
    arr_header.append('label_min')
    tmp_array.insert(0, arr_header)

    for learning_rate_rbm, learning_rate, number_visible_input, number_visible_hidden, \
        mse_train, mse_test, last_index, hmcr_par1, hmcr_par2, hmcr_par3, hmcr_par4, label_replace, label_max, label_min \
            in tmp_array:
        worksheet.write(row, col, row)
        worksheet.write(row, col + 1, learning_rate_rbm)
        worksheet.write(row, col + 2, learning_rate)
        worksheet.write(row, col + 3, number_visible_input)
        worksheet.write(row, col + 4, number_visible_hidden)
        worksheet.write(row, col + 5, mse_train)
        worksheet.write(row, col + 6, mse_test)
        worksheet.write(row, col + 7, last_index)
        worksheet.write(row, col + 8, hmcr_par1)
        worksheet.write(row, col + 9, hmcr_par2)
        worksheet.write(row, col + 10, hmcr_par3)
        worksheet.write(row, col + 11, hmcr_par4)
        worksheet.write(row, col + 12, label_replace)
        worksheet.write(row, col + 13, label_max)
        worksheet.write(row, col + 14, label_min)
        row += 1
    workbook.close()


# export_result('result_hs_memory', result_data)
export_result('result_hs_yearssn', TensorGlobal.followHs)
