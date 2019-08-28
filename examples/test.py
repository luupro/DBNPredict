import pandas as pd
import numpy as np

data = np.array([['','Col1','Col2'],
                 ['t-1',1,2],
                 ['t-2',3,4]])

print(pd.DataFrame(data=data[1:,1:],
                   index=data[1:,0],
                   columns=data[0,1:]))

data_input = data[['t-1', 't-2']].values

print(data_input)

