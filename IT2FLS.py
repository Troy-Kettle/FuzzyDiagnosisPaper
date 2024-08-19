import numpy as np
import matplotlib.pyplot as plt

# Define the universe of discourse
temperature_universe = np.linspace(15, 30, 100)  # Temperature range: 15°C to 30°C
comfort_universe = np.linspace(18, 28, 100)  # Comfort level: 18°C to 28°C
power_universe = np.linspace(0, 100, 100)  # Cooling power: 0% to 100%

# Define Gaussian membership functions for IT2 fuzzy sets
def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def interval_type_2_fuzzy_set(x, mean, sigma, spread):
    lower_membership = gaussian(x, mean - spread, sigma)
    upper_membership = gaussian(x, mean + spread, sigma)
    return lower_membership, upper_membership

# Define IT2 fuzzy sets for Temperature
low_temp = interval_type_2_fuzzy_set(temperature_universe, 17, 2, 1)
medium_temp = interval_type_2_fuzzy_set(temperature_universe, 22, 2, 1)
high_temp = interval_type_2_fuzzy_set(temperature_universe, 27, 2, 1)

# Define IT2 fuzzy sets for Comfort Level
cool_comfort = interval_type_2_fuzzy_set(comfort_universe, 19, 2, 1)
comfortable_comfort = interval_type_2_fuzzy_set(comfort_universe, 23, 2, 1)
warm_comfort = interval_type_2_fuzzy_set(comfort_universe, 26, 2, 1)

# Define IT2 fuzzy sets for Cooling Power
low_power = interval_type_2_fuzzy_set(power_universe, 20, 10, 10)
medium_power = interval_type_2_fuzzy_set(power_universe, 50, 10, 10)
high_power = interval_type_2_fuzzy_set(power_universe, 80, 10, 10)

# Function to compute the membership value at a specific input
def get_membership(input_value, fuzzy_set, universe):
    input_idx = np.searchsorted(universe, input_value)
    lower_membership = fuzzy_set[0]
    upper_membership = fuzzy_set[1]
    lower_value = lower_membership[input_idx] if input_idx < len(lower_membership) else 0
    upper_value = upper_membership[input_idx] if input_idx < len(upper_membership) else 0
    return lower_value, upper_value

# Simple IT2 FLS Inference Function
def it2fls_inference(temp, comfort):
    # Get the membership values for temperature and comfort level
    temp_low, temp_high = get_membership(temp, (low_temp[0], low_temp[1]), temperature_universe)
    temp_med, temp_high_med = get_membership(temp, (medium_temp[0], medium_temp[1]), temperature_universe)
    temp_high, temp_high_high = get_membership(temp, (high_temp[0], high_temp[1]), temperature_universe)
    
    comfort_cool, comfort_cool_high = get_membership(comfort, (cool_comfort[0], cool_comfort[1]), comfort_universe)
    comfort_comfortable, comfort_comfortable_high = get_membership(comfort, (comfortable_comfort[0], comfortable_comfort[1]), comfort_universe)
    comfort_warm, comfort_warm_high = get_membership(comfort, (warm_comfort[0], warm_comfort[1]), comfort_universe)
    
    # Calculate the firing strengths for each rule
    strength1 = min(temp_high, comfort_cool)
    strength2 = min(temp_med, comfort_comfortable)
    strength3 = min(temp_low, comfort_warm)
    
    # Calculate the defuzzified output
    numerator = (strength1 * np.mean(high_power[1]) + 
                 strength2 * np.mean(medium_power[1]) + 
                 strength3 * np.mean(low_power[1]))
    denominator = (strength1 + strength2 + strength3)
    
    return numerator / denominator if denominator != 0 else 0

# Plot Fuzzy Sets with Labels
def plot_fuzzy_sets(universe, lower_membership, upper_membership, title):
    plt.figure(figsize=(12, 8))
    plt.plot(universe, lower_membership, 'b', label='Lower Membership Function', linewidth=2)
    plt.plot(universe, upper_membership, 'r', label='Upper Membership Function', linewidth=2)
    plt.fill_between(universe, lower_membership, upper_membership, color='gray', alpha=0.3)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot all fuzzy sets
plot_fuzzy_sets(temperature_universe, low_temp[0], low_temp[1], 'Low Temperature Fuzzy Set')
plot_fuzzy_sets(temperature_universe, medium_temp[0], medium_temp[1], 'Medium Temperature Fuzzy Set')
plot_fuzzy_sets(temperature_universe, high_temp[0], high_temp[1], 'High Temperature Fuzzy Set')

plot_fuzzy_sets(comfort_universe, cool_comfort[0], cool_comfort[1], 'Cool Comfort Level Fuzzy Set')
plot_fuzzy_sets(comfort_universe, comfortable_comfort[0], comfortable_comfort[1], 'Comfortable Comfort Level Fuzzy Set')
plot_fuzzy_sets(comfort_universe, warm_comfort[0], warm_comfort[1], 'Warm Comfort Level Fuzzy Set')

plot_fuzzy_sets(power_universe, low_power[0], low_power[1], 'Low Cooling Power Fuzzy Set')
plot_fuzzy_sets(power_universe, medium_power[0], medium_power[1], 'Medium Cooling Power Fuzzy Set')
plot_fuzzy_sets(power_universe, high_power[0], high_power[1], 'High Cooling Power Fuzzy Set')

# Example Input: Current temperature = 24°C, Desired comfort level = 23°C
current_temp = 24
desired_comfort = 23

# Perform inference
cooling_power = it2fls_inference(current_temp, desired_comfort)
print(f"Recommended Cooling Power: {cooling_power:.2f}%")
