import time

import scipy as sp
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
iris_dataset = load_iris() 

start = time.time()

print("Keys of iris_dataset:\n", iris_dataset.keys())

print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names:", iris_dataset['target_names'])
print("\nFeature names:\n", iris_dataset['feature_names'])
print("\nType of data:", type(iris_dataset['data']))
print("\nShape of data:",iris_dataset['data'].shape)
print("\nFirst five rows of data:\n", iris_dataset['data'][:5])
print("\nType of target:", type(iris_dataset['target']))
print("n\Target:\n", iris_dataset['target'])







#  measuring the pathetic performance....
end = time.time() 
print("\nExecution time for this program is :", 
        (end-start) * 10**3, "ms")