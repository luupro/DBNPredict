import time
start_time =  time.time()

a = 0
for i in range(0, 100000000):
    a = a +i

stop_time = time.time()

print("time process is: " + str(stop_time-start_time))