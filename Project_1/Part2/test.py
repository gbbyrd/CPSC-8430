import csv

list = [[1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],]

# list = [1, 2]


# Import writer class from csv module
from csv import writer
 
# List that we want to add as a new row
List = [6, 'William', 5532, 1, 'UAE']
 
# Open our existing CSV file in append mode
# Create a file object for this file
with open('event.csv', 'a') as f_object:
 
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)
 
    # Pass the list as an argument into
    # the writerow()
    writer_object.writerows(list)
 
    # Close the file object
    f_object.close()
    
