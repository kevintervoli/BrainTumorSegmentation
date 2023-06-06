import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('../BrainTumorSegmentation/VNet/data_VNet_50.csv')

# Specify distinct colors for each line
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Select the columns you're interested in
x = df['epoch']
y_columns =['binary_accuracy','loss','lr','precision','recall','val_binary_accuracy','val_loss','val_precision','val_recall']

# Create a new figure with a specified DPI (for quality)
plt.figure(dpi=300, figsize=[6, 4])

# Plot the data
for i, column in enumerate(y_columns):
    y = df[column]

    # If less than 10 data points, interpolate
    if len(y) < 10:
        x_new = np.linspace(x.min(), x.max(), 10)
        y_new = np.interp(x_new, x, y)
    else:
        x_new, y_new = x, y

    color = colors[i % len(colors)]  # Use modulo operator to loop through available colors
    plt.plot(x_new, y_new, label=column, color=color, alpha=0.5)

# Set the titles and labels
plt.title('VNet 50 Epochs Data Graph')
plt.xlabel('Epoch')
plt.ylabel('Y-axis label')

# Customize ticks and grid
plt.grid(True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Add a legend
plt.legend()

# Save the figure
plt.savefig('VNet_Data_Graph_50_Epochs.png', bbox_inches='tight')

# Display the graph
plt.show()
