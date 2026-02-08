import numpy as np 
# sparse matrices from scipy provide 2d arrays
# of mostly 0 values
from scipy import sparse

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display



x = np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))


# create a 2D array in numpy w a diag of 1's
# and 0's everywhwere else 

eye = np.eye(4)
print("Numpy array:\n", eye)

eye = np.eye(3)
print("Numpy array:\n", eye)

# conver the numpy array to a scipy sparse matrix 
# in CSR format
#CSR = compressed sparse row matrix 
# only the nonzero entries are stored

sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation: \n", eye_coo)


# generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10,10,100)
# create a second array using sine 
y = np.sin(x)
# the plot function makes a line chart of one array against another
plt.plot(x,y, marker = "x") 
plt.show()


# create a simple dataset of people 
data = {'Name': ["John", "Anna", "Peter", "Linda"],
    'Location' : ["New York", "Paris", "Berlin", "London"],
    'Age' : [24, 13, 53, 33]
    }
data_pandas = pd.DataFrame(data)
#IPython.display allows "pretty printing" of dataframes in jup notebooks

from IPython.display import display
display(data_pandas)

# print(data_pandas)

# possible ways to query the table 
# select all rows that have an age column > 30 
# using Ipython display:   display(data_pandas)
print(data_pandas[data_pandas.Age > 30])
