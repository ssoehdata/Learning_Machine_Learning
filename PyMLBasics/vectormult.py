# scalar multiplication example

import numpy as np 
import matplotlib.pyplot as plt 
import math

v = np.array([4,1])
w = 5 * v
print("w = ", w)

#plot w

origin = [0], [0]
plt.grid()
plt.ticklabel_format(style = 'sci', axis = 'both',
                    scilimits = (0, 0))
plt.quiver(*origin, *w, scale = 10)
plt.show()

# dot product multiplication 

v = np.array([2,1])
s = np.array([3, -2])
d = np.dot(v,s)
print("The dot product of d (using d = v@s) is : ", d)

# cross product 

v = np.array([4,9,12])
s = np.array([21,32,44])
r = np.cross(v, s)
print("The cross product of arrays v and s defined as r:", r)

