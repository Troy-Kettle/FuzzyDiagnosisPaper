import numpy as np
import matplotlib.pyplot as plt

# Define universe of discourse
def create_universes():
    universes = {
        'sbp': np.arange(0, 200, 1),
        'hr': np.arange(0, 140, 1),
        'spo2': np.arange(80, 100, 1),
        'temp': np.arange(35, 40, 0.1),
        'bs': np.arange(60, 160, 1),
        'risk': np.arange(0, 14, 0.5)
    }
    return universes

# Create trapezoidal membership function
def trapmf(x, params):
    a, b, c, d = params
    y = np.zeros_like(x)
    y[x <= a] = 0
    y[(a < x) & (x < b)] = (x[(a < x) & (x < b)] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1
    y[(c < x) & (x < d)] = (d - x[(c < x) & (x < d)]) / (d - c)
    y[x >= d] = 0
    return y

# Create interval type-2 fuzzy set
def create_it2fs(universe, params, uncertainty=0.1):
    lower_params = [p - uncertainty for p in params]
    upper_params = [p + uncertainty for p in params]
    lower_mf = trapmf(universe, lower_params)
    upper_mf = trapmf(universe, upper_params)
    return lower_mf, upper_mf

# Define fuzzy sets for each input
def define_fuzzy_sets(universes):
    fuzzy_sets = {
        'sbp': {
            'Low +3': create_it2fs(universes['sbp'], [0, 0, 70, 80]),
            'Low +2': create_it2fs(universes['sbp'], [70, 80, 85, 90]),
            'Low +1': create_it2fs(universes['sbp'], [85, 90, 100, 110]),
            'Normal +0': create_it2fs(universes['sbp'], [100, 110, 180, 185]),
            'High +2': create_it2fs(universes['sbp'], [180, 185, 200, 200])
        },
        'hr': {
            'Low +2': create_it2fs(universes['hr'], [0, 0, 45, 55]),
            'Low +1': create_it2fs(universes['hr'], [45, 55, 60, 65]),
            'Normal +0': create_it2fs(universes['hr'], [60, 65, 85, 95]),
            'High +1': create_it2fs(universes['hr'], [85, 95, 105, 115]),
            'High +2': create_it2fs(universes['hr'], [105, 115, 125, 130]),
            'High +3': create_it2fs(universes['hr'], [125, 130, 140, 140])
        },
        'spo2': {
            'Low +3': create_it2fs(universes['spo2'], [80, 80, 83, 85]),
            'Low +2': create_it2fs(universes['spo2'], [83, 85, 88, 90]),
            'Low +1': create_it2fs(universes['spo2'], [88, 90, 93, 95]),
            'Normal +0': create_it2fs(universes['spo2'], [93, 95, 100, 100])
        },
        'temp': {
            'Low +2': create_it2fs(universes['temp'], [35, 35, 36, 37]),
            'Normal +0': create_it2fs(universes['temp'], [36, 37, 38, 39]),
            'High +2': create_it2fs(universes['temp'], [38, 39, 40, 40])
        },
        'bs': {
            'Low +3': create_it2fs(universes['bs'], [60, 60, 63, 68]),
            'Low +2': create_it2fs(universes['bs'], [63, 68, 75, 80]),
            'Normal +0': create_it2fs(universes['bs'], [75, 80, 110, 115]),
            'High +2': create_it2fs(universes['bs'], [110, 115, 140, 150]),
            'High +3': create_it2fs(universes['bs'], [140, 150, 160, 160])
        },
        'risk': {
            'NRM': create_it2fs(universes['risk'], [0, 0, 0.5, 1]),
            'LRG1': create_it2fs(universes['risk'], [0.5, 1, 1.5, 2]),
            'LRG2': create_it2fs(universes['risk'], [1.5, 2, 2.5, 3]),
            'LRG3': create_it2fs(universes['risk'], [2.5, 3, 3.5, 4]),
            'LRG4': create_it2fs(universes['risk'], [3.5, 4, 4.5, 5]),
            'HRG5': create_it2fs(universes['risk'], [4.5, 5, 5.5, 6]),
            'HRG6': create_it2fs(universes['risk'], [5.5, 6, 6.5, 7]),
            'HRG7': create_it2fs(universes['risk'], [6.5, 7, 7.5, 8]),
            'HRG8': create_it2fs(universes['risk'], [7.5, 8, 8.5, 9]),
            'HRG9': create_it2fs(universes['risk'], [8.5, 9, 9.5, 10]),
            'HRG10': create_it2fs(universes['risk'], [9.5, 10, 10.5, 11]),
            'HRG11': create_it2fs(universes['risk'], [10.5, 11, 11.5, 12]),
            'HRG12': create_it2fs(universes['risk'], [11.5, 12, 12.5, 13]),
            'HRG13': create_it2fs(universes['risk'], [12.5, 13, 13.5, 14]),
            'HRG14': create_it2fs(universes['risk'], [13.5, 14, 14, 14])
        }
    }
    return fuzzy_sets

# Evaluate fuzzy rules
def evaluate_rules(inputs, fuzzy_sets):
    # Placeholder: Implement your rule-based logic here
    return fuzzy_sets['risk']['NRM']  # Defaulting to 'NRM' for simplicity

# Compute fuzzy risk based on inputs
def compute_risk(input_values, fuzzy_sets):
    fuzzy_risk = evaluate_rules(input_values, fuzzy_sets)
    return fuzzy_risk

# Plot interval type-2 fuzzy sets
def plot_it2fs(ax, universe, it2fs_dict, title):
    for label, (lower_mf, upper_mf) in it2fs_dict.items():
        ax.plot(universe, lower_mf, linestyle='--', alpha=0.8, label=f'{label} (Lower Bound)')
        ax.plot(universe, upper_mf, linestyle='-', alpha=0.8, label=f'{label} (Upper Bound)')
        ax.fill_between(universe, lower_mf, upper_mf, alpha=0.3)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Membership', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def main():
    universes = create_universes()
    fuzzy_sets = define_fuzzy_sets(universes)

    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Interval Type-2 Fuzzy Sets for MEWS-based Categorization', fontsize=16)

    plot_it2fs(axs[0, 0], universes['sbp'], fuzzy_sets['sbp'], 'Systolic Blood Pressure (SBP)')
    plot_it2fs(axs[0, 1], universes['hr'], fuzzy_sets['hr'], 'Heart Rate (HR)')
    plot_it2fs(axs[1, 0], universes['spo2'], fuzzy_sets['spo2'], 'SPO2')
    plot_it2fs(axs[1, 1], universes['temp'], fuzzy_sets['temp'], 'Temperature (T)')
    plot_it2fs(axs[2, 0], universes['bs'], fuzzy_sets['bs'], 'Blood Sugar (BS)')
    plot_it2fs(axs[2, 1], universes['risk'], fuzzy_sets['risk'], 'Risk')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to accommodate the legend
    plt.show()

if __name__ == '__main__':
    main()
