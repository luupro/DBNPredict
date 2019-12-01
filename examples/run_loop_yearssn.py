import numpy as np
import xlsxwriter

import time
from random import *
import tensorflow as tf

from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from dbn.utils import read_file
from examples.RandomRegression import RandomRegression
from examples.TensorGlobal import TensorGlobal
from datetime import datetime
from examples.HSMemory import HSMemory

start_begin = time.time()
path = 'chaotic-timeseries/yearssn.txt'
xs = np.array(read_file(path))

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))

cost_function_name = 'mse'
min_mse = 1000
arr_train_loss = []

result_data = []

best_lrr = 0
best_lr = 0
data_train, label_train, data_test, label_test = None, None, None, None

for i in range(0, 1000):
    print('Run index: %f' % i)
    RandomRegression.number_visible_input = randint(1, 10)
    RandomRegression.number_hidden_input = randint(1, 10)
    data_train, label_train, data_test, label_test = \
        HSMemory.create_train_and_test_data(lorenz_scale.tolist(), RandomRegression.number_visible_input)
    print("Shape of data_train: " + str(data_train.shape))
    print("shape of label_train: " + str(label_train.shape))

    start_time = time.time()
    regressor, tmp_lrr, tmp_lr = RandomRegression.create_random_model()
    regressor.fit(data_train, label_train)
    tmp_train_mse = sum(regressor.train_loss) / RandomRegression.number_iter_backprop
    tmp_min_mse_label = 'MORE BAD'
    stop_time = time.time()
    print("THE TIME FOR TRAINING: " + str((stop_time - start_time)) + ' second')

    # Test
    start_time = time.time()
    Y_pred_test = regressor.predict(data_test)
    print("Begin to call mean_squared_error")
    check_nan = np.isnan(Y_pred_test).any()

    if check_nan:
        tmp_test_mse = 1000
    else:
        tmp_test_mse = mean_squared_error(label_test, Y_pred_test)
    if np.isnan(tmp_train_mse) or np.isinf(tmp_train_mse):
        tmp_train_mse = 1000

    stop_time = time.time()
    print("THE TIME FOR TEST: " + str((stop_time - start_time)) + ' second')
    TensorGlobal.sessFlg = True
    tf.reset_default_graph()
    del regressor
    tmp_element_data = [tmp_lrr, tmp_lr, RandomRegression.number_visible_input, RandomRegression.number_hidden_input,
                        tmp_train_mse, tmp_test_mse]
    result_data.append(tmp_element_data)

# Export result to excel file
print("Begin to print result")
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M%S")
workbook = xlsxwriter.Workbook('result_yearssn_' + dt_string + '.xlsx')
worksheet = workbook.add_worksheet()
row = 1
col = 0

for learning_rate_rbm, learning_rate, number_visible_input, number_visible_hidden, mse_train, mse_test in result_data:
    worksheet.write(row, col, learning_rate_rbm)
    worksheet.write(row, col + 1, learning_rate)
    worksheet.write(row, col + 2, number_visible_input)
    worksheet.write(row, col + 3, number_visible_hidden)
    worksheet.write(row, col + 4, mse_train)
    worksheet.write(row, col + 5, mse_test)
    row += 1
workbook.close()

start_end = time.time()
print("THE TIME FOR TOTAL: " + str((start_end - start_begin)) + ' second')
