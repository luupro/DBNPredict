from Particle.Particle import Particle
from Particle.Space import Space
from datetime import datetime
import numpy as np
import time
from dbn.utils import read_file
from sklearn.preprocessing import MinMaxScaler
import xlsxwriter
from TensorGlobal import TensorGlobal

# Read file process start
start_begin = time.time()

path = 'chaotic-timeseries/yearssn.txt'  # vdmt
xs = np.array(read_file(path))

xs = xs.reshape(-1, 1)
minmax = MinMaxScaler().fit(xs.astype('float32'))
lorenz_scale = minmax.transform(xs.astype('float32'))
# Read file process end


n_iterations = int(100)
target_error = 0.000000001
n_particles = 10

search_space = Space(1, target_error, n_particles)
particles_vector = []
for i in range(0, search_space.n_particles):
    tmpParticle = Particle(lorenz_scale.tolist(), search_space.n_particles, i)
    particles_vector.append(tmpParticle)

search_space.particles = particles_vector
search_space.print_particles()

iteration = int(0)
while(iteration < n_iterations):
    print("iteration " + str(iteration))
    search_space.set_pbest_gbest()
    #search_space.set_gbest()

    # ignore target_error
    #if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
    #break

    search_space.move_particles()
    iteration += 1

#print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)


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
    arr_header.append('label_particle')
    arr_header.append('pbest_value_mse')
    arr_header.append('gbest_value_mse')
    arr_header.append('')
    arr_header.append('')
    arr_header.append('')
    arr_header.append('')
    arr_header.append('')
    tmp_array.insert(0, arr_header)

    for learning_rate_rbm, learning_rate, number_visible_input, number_visible_hidden, \
        mse_train, mse_test, label_particle, pbest_value_mse, gbest_value_mse, x3, x4, x5, x6, x7 \
            in tmp_array:
        worksheet.write(row, col, row)
        worksheet.write(row, col + 1, learning_rate_rbm)
        worksheet.write(row, col + 2, learning_rate)
        worksheet.write(row, col + 3, number_visible_input)
        worksheet.write(row, col + 4, number_visible_hidden)
        worksheet.write(row, col + 5, mse_train)
        worksheet.write(row, col + 6, mse_test)
        worksheet.write(row, col + 7, label_particle)
        worksheet.write(row, col + 8, pbest_value_mse)
        worksheet.write(row, col + 9, gbest_value_mse)
        row += 1
    workbook.close()


# export_result('result_hs_memory', result_data)

export_result('result_pso_yearssn_', TensorGlobal.followHs)  # vdmt

start_end = time.time()
print("THE TIME FOR TOTAL: " + str((start_end - start_begin)) + ' second')