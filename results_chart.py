import csv
import matplotlib.pyplot as plt
import os

def plot_graph(x_data, y_data, title, x_label, y_label, save_path):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()  # Close the figure after saving

def read_csv_file(file_path):
    data = []
    headers = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader, [])  # Read and store the header row or use an empty list as default

        for row in csv_reader:
            data.append([float(val) for val in row])

    return headers, list(zip(*data))  # Convert zip object to list and return headers

# Directory path containing the CSV files
csv_directory = '../BrainTumorSegmentation/Thesis_Results'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

# Read and plot data from each CSV file
for file_name in csv_files:
    # Path to the current CSV file
    csv_file_path = os.path.join(csv_directory, file_name)

    # Read data from the CSV file
    headers, data = read_csv_file(csv_file_path)  # Get headers and transposed data

    # Print the data on the console
    for i, column in enumerate(data):
        print(f"Y-axis data ({file_name}):")
        print(column)

        # Get the header for the current column or use a default label
        y_label = headers[i] if i < len(headers) else f"Column {i+1}"

        # Plot the graph for each column
        save_path = f"../BrainTumorSegmentation/Images/graph_{file_name}_{i+1}.png"  # File path to save the image
        plot_graph(list(data[0]), list(column), title='EPOCHS AND COLUMN', x_label='EPOCH NUMBER', y_label=y_label, save_path=save_path)
