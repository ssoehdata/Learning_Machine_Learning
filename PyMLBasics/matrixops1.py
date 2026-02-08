# demonstrate addition , subtraction and division
# demonstrate multiplication and dot products of matrices
# demonstrate square root, transposition and summation of matrices

# of matrices with numpy

# add(), subtract(), divide() 
# multiply(), dot()
# sqrt(), sum(), "T" to transpose

import numpy as np 

# initialize matrices
x = np.array([[1,2],[4,5]])
y = np.array([[7,8], [9,10]])

# add the matrices
print("The element wise addition of matrix is : ")
print(np.add(x,y))

# matrix subtraction 
print("The element wise subtraction of matrix is : ")
print(np.subtract(x,y))

# matrix division 
print("The element wise division of matrix is : ")
print(np.divide(x,y))


# multiplication 
print("The element wise multiplication of matrix is : ")
print(np.multiply(x,y))

# dot product 
print("The dot product of  matrices is : ")
print(np.dot(x,y))

# square root
print("The element wise square root of matrix is : ")
print(np.sqrt(x))

# using sum to print summation of all elements of matrix
print("The summation of all matrix elements is : ")
print(np.sum(y))

# using sum(axis=0) to print summation of all columns of matrix
print("The column wise summation of all matrix elements is : ")
print(np.sum(y,axis=0))

# using sum(axis=1) to print summation of all columns of matrix
print("The row wise summation of all matrix elements is : ")
print(np.sum(y,axis=1))

# using "T" tp transpose the matrix
print("The transpose of given matrix is : ")
print(x.T)

