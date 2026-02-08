# list comprehension example 
fruits = ['apple', 'banana', 'cherry', 'kiwi', 'mango']
newlist = []

# change the value of the letter below to vary 
# vary the output
for x in fruits:
    if "w" in x:
        newlist.append(x)
print(newlist)