import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
from dbn.utils import series_to_supervised, split_data, mean_absolute_percentage_error, read_file
from examples.RandomRegression import RandomRegression

path = 'chaotic-timeseries/Lorenz3.txt'
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

start_time = time.time()

main_regression = SupervisedDBNRegression()
min_mse = 1000
arr_train_loss = []

result_data = []
result_label = ('learning_rate_rbm', 'learning_rate', 'mse_train', 'mse_test', 'min_mse_label')
best_lrr = 0
best_lr = 0
for i in range(0, 3):
    regressor, tmp_lrr, tmp_lr = RandomRegression.create_random_model()
    regressor.fit(data_train, label_train)
    tmp_train_mse = sum(regressor.train_loss) / 1500
    tmp_min_mse_label = 'MORE BAD'

    # Test
    Y_pred_test = regressor.predict(data_test)
    tmp_test_mse = mean_squared_error(label_test, Y_pred_test)

    if tmp_test_mse < min_mse:
        min_mse = tmp_test_mse
        best_lrr = tmp_lrr
        best_lr = tmp_lr
        tmp_min_mse_label = 'MORE WELL'
    else:
        del regressor
    tmp_element_data = [tmp_lrr, tmp_lr, tmp_train_mse, tmp_test_mse, tmp_min_mse_label]
    result_data.append(tmp_element_data)

'''
main_regression = SupervisedDBNRegression(hidden_layers_structure=[5],
                                        learning_rate_rbm=0.01,
                                        learning_rate=0.03,
                                        n_epochs_rbm=100,
                                        n_iter_backprop=1500,
                                        contrastive_divergence_iter=2,
                                        batch_size=32,
                                        activation_function='relu',
                                        n_hidden_layers_mlp=1,
                                        cost_function_name='mse')
main_regression.fit(data_train, label_train)
'''

stop_time = time.time()

print("THE TIME FOR TRAINING IS: " + str((stop_time - start_time)) + ' second')

'''
plt.figure(figsize=(8, 5))
plt.plot(main_regression.train_loss, 'r', label='training-loss')
plt.plot(main_regression.test_loss, 'g', label='test-loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.ylim([0, 0.002])
plt.show()
'''

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
print('Done.  \n MSE_TEST: %f' % tmp_test_mse)
tmp_test_mae = mean_absolute_error(label_test, Y_pred_test)
print('Done.  \n MAE_TEST: %f' % tmp_test_mae)
tmp_test_mape = mean_absolute_percentage_error(label_test, Y_pred_test)
print('Done.  \n MAPE_TEST: %f' % tmp_test_mape)
'''
# Print result
fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=result_data, colLabels=result_label, loc='center')

fig.tight_layout()

plt.show()