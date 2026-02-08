import time 
import sklearn
from sklearn.datasets import load_iris
iris_dataset = load_iris() 
import pandas as pd 
import numpy as np
import mglearn



start = time.time()

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],random_state=0)

print("X_train shape:", X_train.shape)
print("\ny_train shape:", y_train.shape)
print("\nX_test shape:", X_test.shape)
print("\ny_test shape:", y_test.shape)

############################
# visualizing data for initial data inspection 
# will use a pair plot 
# first convert array to  a pd dataframe using X_train data 
# label the cols using the strings in the iris_dataset.features_names 
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), 
            marker='o',hist_kwds={'bins':20},s=60,
            alpha=.8,cmap=mglearn.cm3)




# page 21 Intro to ML w Python

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
        metric_params=None, n_jobs=None, n_neighbors=1, p=2, 
        weights='uniform')

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

X_new.shape:(1,4)

prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
    iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


















end = time.time() 
print("\nExecution time for this program is :", 
        (end-start) * 10**3, "ms")