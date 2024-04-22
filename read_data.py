import numpy as np
import csv

# Read the CSV file (replace 'your_file.csv' with the actual file path)
csv_file = 'train_data1.csv'

# Initialize lists to store relevant columns
type_column = []
price_column = []
bath_column = []
propertysqft_column = []

# Process the CSV file using the csv library
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        # Extract relevant columns
        type_column.append(row[1])
        price_column.append(row[2])
        bath_column.append(row[3])
        propertysqft_column.append(row[4])

# Process the TYPE column
unique_types = np.unique(type_column)
type_mapping = {word.split()[0]: idx for idx, word in enumerate(unique_types)}

# Create a new array with numeric TYPE values
numeric_type_column = np.array([type_mapping[word.split()[0]] for word in type_column], dtype=int)

# Create the final NumPy array
final_array = np.column_stack((numeric_type_column, price_column, bath_column, propertysqft_column))

# Print the final array (you can save it to a file if needed)
print(final_array)




# Read the CSV file with a single column (replace 'your_single_column.csv' with the actual file path)
csv_file = 'train_label1.csv'

# Initialize an empty list to store the values from the single column
bed_values = []

# Process the CSV file using the csv library
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        # Assuming the CSV file has only one column, append the value to the list
        bed_values.append(row[0])

# Create a 1D NumPy array from the single column values
beds = np.array(bed_values)

# Print the resulting array (you can save it to a file if needed)
print(beds)


