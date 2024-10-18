import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Function to read data from file
def read_data(filename):
    data = defaultdict(lambda: defaultdict(int))
    with open(filename, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            label, tasks, percent = line.strip().split(',')
            data[label.strip()][int(tasks)] = float(percent)
    return dict(data)

# Read data from file
filename = "PercentWithinAllocationTolerance.txt"
data = read_data(filename)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of bars and positions of the bars on the x-axis
bar_width = 0.2
labels = list(data.keys())
x = np.arange(len(labels))

# Create bars and add labels on top
for i, num_tasks in enumerate([4, 6, 8]):
    values = [data[label].get(num_tasks, 0) for label in labels]
    bars = plt.bar(x + i*bar_width, values, width=bar_width, label=f'{num_tasks} tasks')
    
    # Add labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=14)

# Add labels and title with increased font size
plt.xlabel('Model', fontsize=18)
plt.ylabel('Percent Within 15% of Optimal (%)', fontsize=18)
plt.title('Percent Within 15% of Optimal Solution by LLM and Number of Tasks', fontsize=18)
plt.xticks(x + bar_width, labels, fontsize=14)
plt.yticks(fontsize=16)

# Create legend
plt.legend(fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()