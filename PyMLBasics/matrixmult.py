# can use list comprehension to multiply matrices

# a 3x3 matrix example 
A = [[12, 7, 3],
    [4,5,6],
    [7,8,9]]

# a 3x4 matrix for B matrix

B = [[5,8,1,2],
    [6,7,3.0],
    [4,5,9,1]]

# which will result in a 3x4 matrix
# this method makes use of the zip function 
result = [[sum(a * b for a, b in zip(A_row, B_col))
                for B_col in zip(*B)]
                        for A_row in A]
for r in result:
    print(r)
