import numpy as np
import matplotlib.pyplot as plt
#  import logging

from examples.HSMemory import HSMemory
from examples.HSElement import HSElement
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import series_to_supervised, split_data, read_file, mean_absolute_percentage_error
from random import *

# For log
#  logger = logging.getLogger().addHandler(logging.StreamHandler())


def transfer_hs_memory_data(tmp_hs_memory):  # print HS
    tmp_all_data = []
    tmp_header_data = ('learning_rate_rbm', 'learning_rate', 'mse_train', 'mse_test')
    for ii in range(0, len(tmp_hs_memory)):
        tmp1_hs_element = tmp_hs_memory[ii]
        tmp_lrr = tmp1_hs_element.learning_rate_rbm
        tmp_lr = tmp1_hs_element.learning_rate
        tmp_mse = tmp1_hs_element.train_lost
        tmp1_element_data = [tmp_lrr, tmp_lr, tmp_mse, 'No Count']
        tmp_all_data.append(tmp1_element_data)
    return tmp_all_data, tmp_header_data
# --------------------------------------------


path = 'chaotic-timeseries/Lorenz3.txt'
xs = np.array(read_file(path))
'''
plt.plot(xs)
plt.ylabel('x(t)')
plt.title("Thousand point Lorenz Chaos Time Series")
plt.show()
'''
xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))

data = series_to_supervised(lorenz_scale.tolist(), 5)

data_input = data[['t-5', 't-4', 't-3', 't-2', 't-1']].values  # convert Dataframe to numpy array
data_labels = data['t'].values.reshape(-1, 1)

data_train, label_train, data_test, label_test = split_data(data_input, data_labels)

print("Shape of data_train: " + str(data_train.shape))
print("shape of label_train: " + str(label_train.shape))

cost_function_name = 'mse'
# cost_function_name = 'mae'
# cost_function_name = 'mape'

# Loop for
element = HSElement()
hs_memory_object = HSMemory(data_train, label_train)
main_regression = SupervisedDBNRegression()
NI = 4  # number of improvisations


# Improve a new hamony
new_harmony_element = HSElement()  # already random for all variable

result_data, result_label = transfer_hs_memory_data(HSMemory.hmMemory)
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=result_data, colLabels=result_label, loc='center')
fig.tight_layout()
plt.title("Table HS After Init")
plt.show()

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
            '''
            if i == 3:
                new_harmony_element.n_epochs_rbm = tmp_hm_element.n_epochs_rbm
            if i == 4:
                new_harmony_element.n_iter_back_prop = tmp_hm_element.n_iter_back_prop
            if i == 5:
                new_harmony_element.contrastive_divergence_iter = tmp_hm_element.contrastive_divergence_iter
            '''
    # Step4 Update harmony memory
    print('Update train mse  for Improve harmony h: %f' % h)
    new_harmony_element = HSMemory. \
        update_train_lost(new_harmony_element, data_train, label_train)
    # compare with worst element
    if new_harmony_element.train_lost < HSMemory.max_train_lost:
        HSMemory.hmMemory[HSMemory.max_index] = new_harmony_element  # replace worst
        print('replace max_index: %f' % HSMemory.max_index)
        print('new train_lost: %f' % new_harmony_element.train_lost)
        if HSMemory.min_train_lost > new_harmony_element.train_lost:
            HSMemory.min_train_lost = new_harmony_element.train_lost # update min train_lost
            HSMemory.min_index = HSMemory.max_index  # update min_index
            print('NEw min_index: %f' % HSMemory.min_index)
        HSMemory.update_max_index()  # update max_index
        print('NEw max_index: %f' % HSMemory.max_index)
# Test
tmp_min_element = HSMemory.hmMemory[HSMemory.min_index]
main_regression = HSMemory. \
                    train_data_and_return_model(tmp_min_element, data_train, label_train)
#  main_regression.set_data_test(data_test, label_test)
Y_pred_test = main_regression.predict(data_test)

# Change to void error
'''
inverse_label_train = minmax.inverse_transform(label_train.reshape(-1, 1))
inverse_label_test = minmax.inverse_transform(label_test)
inverse_Y_pred_test = minmax.inverse_transform(Y_pred_test)

# Plot for training
plt.plot(inverse_label_train, 'r', label='lorenz original')

plt.legend(loc='upper right')
plt.ylabel('x(t)')
plt.title("Training Loren Chaos Time Series")
plt.show()

# Plot for testing
plt.plot(inverse_label_test, 'r', label='lorenz original')
plt.plot(inverse_Y_pred_test, 'g', label='RBM+MLP prediction')
plt.legend(loc='upper right')
plt.ylabel('x(t)')
plt.title("Testing Loren Chaos Time Series")
plt.show()
'''
tmp_test_mse = mean_squared_error(label_test, Y_pred_test)
print('Done.  \n MSE_TEST: %f' % tmp_test_mse)
tmp_test_mae = mean_absolute_error(label_test, Y_pred_test)
print('Done.  \n MAE_TEST: %f' % tmp_test_mae)
tmp_test_mape = mean_absolute_percentage_error(label_test, Y_pred_test)
print('Done.  \n MAPE_TEST: %f' % tmp_test_mape)

result_data, result_label = transfer_hs_memory_data(HSMemory.hmMemory)
# get best Harmory
tmp_hs_element = HSMemory.hmMemory[HSMemory.min_index]
tmp_element_data = [tmp_hs_element.learning_rate_rbm,
                    tmp_hs_element.learning_rate, tmp_hs_element.train_lost, tmp_test_mse]
result_data.append(tmp_element_data)
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=result_data, colLabels=result_label, loc='center')
fig.tight_layout()
plt.title("Table HS After Improvement")
plt.show()
