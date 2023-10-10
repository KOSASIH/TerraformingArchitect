def calculate_resource_requirements(size, composition, atmospheric_conditions):
    """
    Calculate the required resources for terraforming a celestial body based on specific parameters.
    
    Parameters:
    - size: The size of the celestial body (e.g., radius in kilometers).
    - composition: The composition of the celestial body (e.g., rocky, gaseous, icy).
    - atmospheric_conditions: The desired atmospheric conditions (e.g., oxygen level, greenhouse gases).
    
    Returns:
    - Markdown table with the resource requirements for each parameter combination.
    """
    
    # Define the resource requirements based on the parameter combinations
    resource_requirements = []
    
    if composition == "rocky":
        resource_requirements.append(["Size", "Composition", "Atmospheric Conditions", "Materials", "Energy"])
        
        if atmospheric_conditions == "oxygen-rich":
            materials = size * 1000  # Example calculation for materials requirement
            energy = size * 100  # Example calculation for energy requirement
            resource_requirements.append([size, composition, atmospheric_conditions, materials, energy])
        
        elif atmospheric_conditions == "nitrogen-rich":
            materials = size * 500  # Example calculation for materials requirement
            energy = size * 50  # Example calculation for energy requirement
            resource_requirements.append([size, composition, atmospheric_conditions, materials, energy])
        
        # Add more conditions and calculations based on other atmospheric conditions if needed
        
    elif composition == "gaseous":
        resource_requirements.append(["Size", "Composition", "Atmospheric Conditions", "Materials", "Energy"])
        
        # Add calculations for resource requirements based on gaseous composition
        
    elif composition == "icy":
        resource_requirements.append(["Size", "Composition", "Atmospheric Conditions", "Materials", "Energy"])
        
        # Add calculations for resource requirements based on icy composition
    
    # Add more conditions and calculations based on other composition types if needed
    
    # Convert the resource requirements into a markdown table
    markdown_table = "|".join(resource_requirements[0]) + "\n"
    markdown_table += "|".join(["---"] * len(resource_requirements[0])) + "\n"
    
    for row in resource_requirements[1:]:
        markdown_table += "|".join(map(str, row)) + "\n"
    
    return markdown_table
