from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt 
from matplotlib import style 

style.use("fivethirtyeight")

X, y = make = make_blobs(n_samples = 100, centers = 3,
                        cluster_std = 1, n_features = 2)

plt.scatter(X[:, 0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()


from sklearn.datasets import make_moons
X,y = make_moons(n_samples = 1000, noise = 0.1)
plt.scatter(X[:,0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()

#### make circles
from sklearn.datasets import make_circles 
style.use("fivethirtyeight")

X,y = make_circles(n_samples = 100, noise = 0.02)
plt.scatter(X[:,0], X[:, 1], s = 40, color = 'g')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()




