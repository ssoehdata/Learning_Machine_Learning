# example of a synthetic 2 class classification dataset (forge dataset)


import mglearn
import matplotlib.pyplot as plt
# generate the dataset
X, y = mglearn.datasets.make_forge() 
#plot the dataset 
mglearn.discrete_scatter(X[:, 0], X[:,1],y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:", X.shape)
plt.show()



