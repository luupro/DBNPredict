import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import series_to_supervised, split_data, mean_absolute_percentage_error, read_file



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



regressor = SupervisedDBNRegression(hidden_layers_structure=[5],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.03,
                                    n_epochs_rbm=100,
                                    n_iter_backprop=1500,
                                    contrastive_divergence_iter=2,
                                    batch_size=32,
                                    activation_function='relu',
                                    n_hidden_layers_mlp=0,
                                    cost_function_name=cost_function_name,
                                    )

regressor.set_data_test(data_test, label_test)

start_time = time.time()

regressor.fit(data_train, label_train)

stop_time = time.time()




print("THE TIME FOR TRAINING IS: " + str((stop_time - start_time)) + ' second')

plt.figure(figsize=(8, 5))
plt.plot(regressor.train_loss, 'r', label='training-loss')
plt.plot(regressor.test_loss, 'g', label='test-loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.ylim([0, 0.002])
plt.show()

# Test
Y_pred_test = regressor.predict(data_test)
Y_pred_train = regressor.predict(data_train)

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
