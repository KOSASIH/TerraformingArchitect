import numpy as np
import matplotlib.pyplot as plt

def simulate_organism_growth(organism_population, resource_availability, competition_factor, generations):
    """
    Simulates the growth and evolution of introduced organisms on a terraformed celestial body.
    
    Args:
    - organism_population (float): Initial population of the introduced organisms.
    - resource_availability (float): Availability of resources for the organisms.
    - competition_factor (float): Factor representing the competition for resources.
    - generations (int): Number of generations to simulate.
    
    Returns:
    - List[float]: List of population sizes for each generation.
    """
    population_sizes = [organism_population]
    
    for _ in range(generations):
        population = population_sizes[-1]
        new_population = population + (resource_availability * population) - (competition_factor * population**2)
        population_sizes.append(new_population)
    
    return population_sizes

# Example usage
initial_population = 1000
resource_availability = 0.8
competition_factor = 0.001
generations = 10

population_sizes = simulate_organism_growth(initial_population, resource_availability, competition_factor, generations)

# Plotting the population sizes over generations
plt.plot(range(generations + 1), population_sizes)
plt.xlabel('Generation')
plt.ylabel('Population Size')
plt.title('Organism Population Growth')
plt.show()
