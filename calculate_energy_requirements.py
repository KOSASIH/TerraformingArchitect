def calculate_energy_requirements(atmospheric_composition, temperature, solar_radiation):
    # Constants
    specific_heat_capacity = 1000  # J/kg·K
    surface_area = 4 * 3.14 * (radius ** 2)  # Assuming a spherical celestial body
    time_period = 24 * 60 * 60  # Time period for which energy requirements are calculated (in seconds)

    # Calculate energy requirements
    energy_requirements = specific_heat_capacity * surface_area * temperature * time_period

    # Adjust energy requirements based on atmospheric composition and solar radiation
    if atmospheric_composition == "oxygen":
        energy_requirements *= 1.2  # Increase energy requirements for oxygen-rich atmospheres
    elif atmospheric_composition == "carbon_dioxide":
        energy_requirements *= 1.5  # Increase energy requirements for carbon dioxide-rich atmospheres

    if solar_radiation == "high":
        energy_requirements *= 1.2  # Increase energy requirements for high solar radiation
    elif solar_radiation == "low":
        energy_requirements *= 0.8  # Decrease energy requirements for low solar radiation

    return energy_requirements
