# demonstrates the zip function  also used 
# in the matrix mutliplication example 

# Two separate lists
cars = ["Aston", "Audi", "McLaren"]
accessories = [ "GPS", "Car Repair Kit",
                "Dolby Sound Kit"] 

# combine the lists using zip function and print them

for c, a in zip(cars, accessories):
    print ("Car: %s, Accesory required: %s"\
        % (c,a))


# the opposite of this function is unzip: 

# Unzip lists
l1,l2 = zip(*[('Aston', 'GPS'), 
              ('Audi', 'Car Repair'), 
              ('McLaren', 'Dolby sound kit') 
           ])
 
# Printing unzipped lists 
print(" *** and these are the lists unzipped  ***")     
print(l1)
print(l2)