import numpy as np
import matplotlib.pyplot as plt

def generate_heat_map(atmospheric_composition, land_cover, solar_radiation):
    # Define temperature ranges based on atmospheric composition
    temperature_ranges = {
        'oxygen': (0, 30),
        'nitrogen': (-10, 20),
        'carbon_dioxide': (-20, 10)
    }
    
    # Define land cover temperature modifiers
    land_cover_modifiers = {
        'water': -5,
        'forest': 2,
        'desert': 5
    }
    
    # Calculate temperature changes based on atmospheric composition
    temperature_change = temperature_ranges[atmospheric_composition]
    
    # Apply land cover temperature modifiers
    temperature_change += land_cover_modifiers[land_cover]
    
    # Apply solar radiation effect
    temperature_change += solar_radiation
    
    # Generate heat map
    heat_map = np.random.uniform(temperature_change[0], temperature_change[1], size=(100, 100))
    
    # Plot heat map
    plt.imshow(heat_map, cmap='hot', origin='lower')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Heat Map of Terraformed Celestial Body')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Example usage
generate_heat_map(atmospheric_composition='nitrogen', land_cover='forest', solar_radiation=10)
