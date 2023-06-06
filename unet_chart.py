import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('../BrainTumorSegmentation/Thesis_Results/data_2.csv')

# Specify distinct colors for each line
colors = ['red', 'blue', 'green', 'black', 'orange', 'purple', 'brown', 'pink', 'gray']

# Select the columns you're interested in
x = df['epoch']
y_columns = ['dice_coef', 'iou', 'loss', 'lr', 'precision', 'recall', 'val_dice_coef', 'val_iou', 'val_loss',
             'val_precision', 'val_recall']

# Create a new figure with a specified DPI (for quality)
fig = plt.figure(dpi=300, figsize=[8, 6])
ax = fig.add_subplot(111, projection='3d')

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
    ax.plot(x_new, y_new, zs=i, label=column, color=color)

# Set the titles and labels
ax.set_title('UNet 50 Epochs Data Graph')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_zlabel('Metric')

# Customize ticks and grid
ax.grid(True)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)

# Add a legend
ax.legend()

# Save the figure
plt.savefig('UNet_Data_3D_Graph.png', bbox_inches='tight')

# Display the graph
plt.show()
