import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for temperature changes
# Replace this with your own data or calculations
temperature_data = np.random.rand(10, 10)

# Create a heat map visualization
plt.imshow(temperature_data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature')
plt.title('Temperature Changes on Terraformed Celestial Body')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Customize the colorbar
colorbar = plt.colorbar()
colorbar.set_label('Temperature')

# Save the visualization to a file
plt.savefig('temperature_heatmap.png')

# Show the visualization
plt.show()
