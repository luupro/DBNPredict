import numpy as np
import matplotlib.pyplot as plt
import logging

from examples.HSMemory import HSMemory
from examples.HSElement import HSElement
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import series_to_supervised, split_data, mean_absolute_percentage_error, read_file
from random import *

# For log
logger = logging.getLogger().addHandler(logging.StreamHandler())

path = 'chaotic-timeseries/Lorenz.txt'
xs = np.array(read_file(path))

plt.plot(xs)
plt.ylabel('x(t)')
plt.title("Thousand point Lorenz Chaos Time Series")
plt.show()

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))

data = series_to_supervised(lorenz_scale.tolist(), 5)
# print(data.head())

data_input = data[['t-5', 't-4', 't-3', 't-2', 't-1']].values  # convert Dataframe to numpy array
data_labels = data['t'].values.reshape(-1, 1)

data_train, label_train, data_test, label_test = split_data(data_input, data_labels)

print("Shape of data_train: " + str(data_train.shape))
print("shape of label_train: " + str(label_train.shape))

cost_function_name = 'mse'
# cost_function_name = 'mae'
# cost_function_name = 'mape'

# Loop for
element = HSElement(False)
hs_memory_object = HSMemory()
# memory = hs_memory_object.get_hs_memory()
main_regression = SupervisedDBNRegression()
NI = 10  # number of improvisations


# Improve a new hamony
new_harmony_element = HSElement(False)  # already random for all variable
for h in range(1, HSMemory.NI):
    for i in range(1, HSMemory.number_decision_var):
        if uniform(0, 1) <= hs_memory_object.HMCR:
            if uniform(0, 1) > hs_memory_object.PAR:
                random_index = random.randint(0, 49)
                tmp_hm_element = HSMemory.hmMemory[random_index]
            else:
                tmp_hm_element = HSMemory.hmMemory[HSMemory.min_index]
            if i == 1:
                new_harmony_element.learning_rate_rbm = tmp_hm_element.learning_rate_rbm
            if i == 2:
                new_harmony_element.learning_rate = tmp_hm_element.learning_rate
            if i == 3:
                new_harmony_element.n_epochs_rbm = tmp_hm_element.n_epochs_rbm
            if i == 4:
                new_harmony_element.n_iter_back_prop = tmp_hm_element.n_iter_back_prop
            if i == 5:
                new_harmony_element.contrastive_divergence_iter = tmp_hm_element.contrastive_divergence_iter
    # Step4 Update harmony memory
    new_harmony_element = HSMemory. \
        train_with_element_option(new_harmony_element, data_train, label_train)
    # compare with worst element
    if HSMemory.max_train_lost > new_harmony_element.train_lost:
        HSMemory.hmMemory[HSMemory.max_index] = new_harmony_element  # replace worst

# Test
Y_pred_test = main_regression.predict(data_test)
Y_pred_train = main_regression.predict(data_train)

# Change to void error
inverse_label_train = minmax.inverse_transform(label_train.reshape(-1, 1))
inverse_label_test = minmax.inverse_transform(label_test)
inverse_Y_pred_train = minmax.inverse_transform(Y_pred_train)
inverse_Y_pred_test = minmax.inverse_transform(Y_pred_test)

# Plot for training
plt.plot(inverse_label_train, 'r', label='lorenz original')
plt.plot(inverse_Y_pred_train, 'g', label='RBM+MLP prediction')

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

print('Done. \n MSE_TRAIN: %f \n MSE_TEST: %f' %
      (mean_squared_error(label_train, Y_pred_train), mean_squared_error(label_test, Y_pred_test)))
print('Done. \n MAE_TRAIN: %f \n MAE_TEST: %f' %
      (mean_absolute_error(label_train, Y_pred_train), mean_absolute_error(label_test, Y_pred_test)))
print('Done.\n MAPE_TRAIN: %f \n MAPE_TEST: %f' % (
    mean_absolute_percentage_error(label_train, Y_pred_train), mean_absolute_percentage_error(label_test, Y_pred_test)))
