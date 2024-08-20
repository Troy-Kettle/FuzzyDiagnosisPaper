import numpy as np
import matplotlib.pyplot as plt

# Define the universe of discourse for each variable
sbp_universe = np.arange(0, 200, 1)
hr_universe = np.arange(0, 140, 1)
spo2_universe = np.arange(80, 100, 1)
temp_universe = np.arange(35, 40, 0.1)
bs_universe = np.arange(60, 160, 1)
risk_universe = np.arange(0, 14, 0.5)

# Function to create trapezoidal membership function
def trapmf(x, params):
    a, b, c, d = params
    y = np.zeros_like(x)
    y[x <= a] = 0
    y[(a < x) & (x < b)] = (x[(a < x) & (x < b)] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1
    y[(c < x) & (x < d)] = (d - x[(c < x) & (x < d)]) / (d - c)
    y[x >= d] = 0
    return y

# Function to create interval type-2 fuzzy set
def create_it2fs(universe, params, uncertainty=0.1):
    lower_params = [p - uncertainty for p in params]
    upper_params = [p + uncertainty for p in params]
    lower_mf = trapmf(universe, lower_params)
    upper_mf = trapmf(universe, upper_params)
    return lower_mf, upper_mf

# Define the Antecedents (Inputs)
sbp = {
    'Low +3': create_it2fs(sbp_universe, [0, 0, 70, 75]),
    'Low +2': create_it2fs(sbp_universe, [70, 75, 80, 85]),
    'Low +1': create_it2fs(sbp_universe, [80, 85, 95, 100]),
    'Normal +0': create_it2fs(sbp_universe, [95, 100, 180, 185]),
    'High +2': create_it2fs(sbp_universe, [180, 185, 200, 200])
}

hr = {
    'Low +2': create_it2fs(hr_universe, [0, 0, 45, 50]),
    'Low +1': create_it2fs(hr_universe, [45, 50, 55, 60]),
    'Normal +0': create_it2fs(hr_universe, [53, 60, 95, 100]),
    'High +1': create_it2fs(hr_universe, [95, 100, 105, 110]),
    'High +2': create_it2fs(hr_universe, [105, 110, 125, 130]),
    'High +3': create_it2fs(hr_universe, [125, 130, 140, 140])
}

spo2 = {
    'Low +3': create_it2fs(spo2_universe, [80, 80, 83, 85]),
    'Low +2': create_it2fs(spo2_universe, [83, 85, 87, 90]),
    'Low +1': create_it2fs(spo2_universe, [87, 90, 92, 95]),
    'Normal +0': create_it2fs(spo2_universe, [93, 95, 100, 100])
}

temp = {
    'Low +2': create_it2fs(temp_universe, [35, 35, 36, 36.5]),
    'Normal +0': create_it2fs(temp_universe, [36, 36.5, 38, 38.5]),
    'High +2': create_it2fs(temp_universe, [38, 38.5, 40, 40])
}

bs = {
    'Low +3': create_it2fs(bs_universe, [60, 60, 63, 66]),
    'Low +2': create_it2fs(bs_universe, [63, 66, 70, 72]),
    'Normal +0': create_it2fs(bs_universe, [70, 72, 106, 110]),
    'High +2': create_it2fs(bs_universe, [106, 110, 140, 150]),
    'High +3': create_it2fs(bs_universe, [140, 150, 160, 160])
}

# Define the Consequent (Output)
risk = {
    'NRM': create_it2fs(risk_universe, [0, 0, 0.5, 1.5]),
    'LRG1': create_it2fs(risk_universe, [0.5, 1.5, 1.5, 2.5]),
    'LRG2': create_it2fs(risk_universe, [1.5, 2.5, 2.5, 3.5]),
    'LRG3': create_it2fs(risk_universe, [2.5, 3.5, 3.5, 4.5]),
    'LRG4': create_it2fs(risk_universe, [3.5, 4.5, 4.5, 5.5]),
    'HRG5': create_it2fs(risk_universe, [4.5, 5.5, 5.5, 6.5]),
    'HRG6': create_it2fs(risk_universe, [5.5, 6.5, 6.5, 7.5]),
    'HRG7': create_it2fs(risk_universe, [6.5, 7.5, 7.5, 8.5]),
    'HRG8': create_it2fs(risk_universe, [7.5, 8.5, 8.5, 9.5]),
    'HRG9': create_it2fs(risk_universe, [8.5, 9.5, 9.5, 10.5]),
    'HRG10': create_it2fs(risk_universe, [9.5, 10.5, 10.5, 11.5]),
    'HRG11': create_it2fs(risk_universe, [10.5, 11.5, 11.5, 12.5]),
    'HRG12': create_it2fs(risk_universe, [11.5, 12.5, 12.5, 13.5]),
    'HRG13': create_it2fs(risk_universe, [12.5, 13.5, 13.5, 14]),
    'HRG14': create_it2fs(risk_universe, [13.5, 14, 14, 14])
}

# Define the fuzzy rules (manually evaluating)
def evaluate_rules(inputs):
    # Placeholder for rule evaluation, return a fuzzy set for risk
    return risk['NRM']  # Defaulting to 'NRM' for simplicity

# Example: Input values
input_values = {
    "SBP": 110,
    "HR": 70,
    "SPO2": 96,
    "Temp": 37.5,
    "BS": 90
}

# Compute the risk
def compute_risk(input_values):
    # Simplified for demonstration, you need to implement rule evaluation
    fuzzy_risk = evaluate_rules(input_values)
    # Aggregate the risk values
    return fuzzy_risk

# Compute the fuzzy output
fuzzy_risk = compute_risk(input_values)

# Function to plot interval type-2 fuzzy sets
def plot_it2fs(ax, universe, it2fs_dict, title):
    for label, (lower_mf, upper_mf) in it2fs_dict.items():
        ax.fill_between(universe, lower_mf, upper_mf, alpha=0.5, label=label)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Membership')

# Create the plot
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Type-2 MEWS-based Categorization Membership Functions', fontsize=16)

plot_it2fs(axs[0, 0], sbp_universe, sbp, 'Systolic Blood Pressure (SBP)')
plot_it2fs(axs[0, 1], hr_universe, hr, 'Heart Rate (HR)')
plot_it2fs(axs[1, 0], spo2_universe, spo2, 'SPO2')
plot_it2fs(axs[1, 1], temp_universe, temp, 'Temperature')
plot_it2fs(axs[2, 0], bs_universe, bs, 'Blood Sugar')
plot_it2fs(axs[2, 1], risk_universe, risk, 'Risk')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
