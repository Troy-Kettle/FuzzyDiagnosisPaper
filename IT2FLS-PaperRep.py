import numpy as np
import matplotlib.pyplot as plt

# Define trapezoidal membership function
def trapezoidal_mf(x, a, b, c, d):
    b = a + 0.01 if a == b else b
    d = c + 0.01 if c == d else d
    return np.maximum(0, np.minimum((x-a)/(b-a), np.minimum(1, (d-x)/(d-c))))

# Define Interval Type-2 trapezoidal membership function
def interval_t2_mf(x, lower, upper):
    lower_mf = trapezoidal_mf(x, *lower)
    upper_mf = trapezoidal_mf(x, *upper)
    return lower_mf, upper_mf

# Define membership functions for Systolic Blood Pressure (SBP)
def sbp_low3(x):
    lower = [50, 50.01, 65, 75]
    upper = [50, 50.01, 70, 80]
    return interval_t2_mf(x, lower, upper)

def sbp_low2(x):
    lower = [70, 75, 80, 85]
    upper = [65, 75, 85, 90]
    return interval_t2_mf(x, lower, upper)

def sbp_low1(x):
    lower = [80, 90, 95, 100]
    upper = [75, 85, 95, 105]
    return interval_t2_mf(x, lower, upper)

def sbp_normal(x):
    lower = [95, 125, 135, 150]
    upper = [90, 120, 140, 155]
    return interval_t2_mf(x, lower, upper)

def sbp_high1(x):
    lower = [135, 145, 165, 185]
    upper = [130, 140, 170, 190]
    return interval_t2_mf(x, lower, upper)

def sbp_high2(x):
    lower = [170, 185, 195, 200]
    upper = [160, 180, 195, 200]
    return interval_t2_mf(x, lower, upper)

def sbp_high3(x):
    lower = [185, 195, 200, 200.01]
    upper = [175, 190, 200, 200.01]
    return interval_t2_mf(x, lower, upper)

# Define membership functions for Heart Rate (HR)
def hr_low3(x):
    lower = [30, 30.01, 40, 50]
    upper = [30, 30.01, 45, 55]
    return interval_t2_mf(x, lower, upper)

def hr_low2(x):
    lower = [45, 50, 55, 60]
    upper = [40, 50, 60, 65]
    return interval_t2_mf(x, lower, upper)

def hr_low1(x):
    lower = [53, 60, 75, 100]
    upper = [50, 60, 80, 105]
    return interval_t2_mf(x, lower, upper)

def hr_normal(x):
    lower = [60, 70, 90, 100]
    upper = [55, 65, 95, 105]
    return interval_t2_mf(x, lower, upper)

def hr_high1(x):
    lower = [90, 100, 115, 125]
    upper = [85, 95, 120, 130]
    return interval_t2_mf(x, lower, upper)

def hr_high2(x):
    lower = [105, 115, 125, 130]
    upper = [100, 110, 130, 135]
    return interval_t2_mf(x, lower, upper)

def hr_high3(x):
    lower = [125, 135, 145, 150]
    upper = [120, 130, 145, 150]
    return interval_t2_mf(x, lower, upper)

# Define membership functions for SPO2
def spo2_low3(x):
    lower = [70, 70.01, 80, 85]
    upper = [70, 70.01, 83, 88]
    return interval_t2_mf(x, lower, upper)

def spo2_low2(x):
    lower = [83, 85, 88, 90]
    upper = [80, 85, 90, 92]
    return interval_t2_mf(x, lower, upper)

def spo2_low1(x):
    lower = [87, 90, 92, 95]
    upper = [85, 88, 93, 97]
    return interval_t2_mf(x, lower, upper)

def spo2_normal(x):
    lower = [93, 95, 98, 100]
    upper = [90, 95, 98, 100]
    return interval_t2_mf(x, lower, upper)

# Define membership functions for Temperature (T)
def t_low3(x):
    lower = [35, 35.01, 36, 36.5]
    upper = [35, 35.01, 36.2, 36.7]
    return interval_t2_mf(x, lower, upper)

def t_low2(x):
    lower = [36.2, 36.5, 36.7, 37]
    upper = [36, 36.5, 37, 37.2]
    return interval_t2_mf(x, lower, upper)

def t_normal(x):
    lower = [36.7, 37, 38.5, 38.5]
    upper = [36.5, 37, 38.5, 38.8]
    return interval_t2_mf(x, lower, upper)

def t_high1(x):
    lower = [38.3, 38.5, 39, 39.5]
    upper = [38, 38.5, 39.2, 40]
    return interval_t2_mf(x, lower, upper)

def t_high2(x):
    lower = [39, 40, 40.5, 41]
    upper = [38.8, 40, 41, 41.01]
    return interval_t2_mf(x, lower, upper)

# Define membership functions for Blood Sugar (BS)
def bs_low3(x):
    lower = [50, 50.01, 60, 66]
    upper = [50, 50.01, 63, 68]
    return interval_t2_mf(x, lower, upper)

def bs_low2(x):
    lower = [63, 66, 70, 72]
    upper = [60, 66, 72, 74]
    return interval_t2_mf(x, lower, upper)

def bs_normal(x):
    lower = [70, 85, 95, 110]
    upper = [65, 80, 100, 115]
    return interval_t2_mf(x, lower, upper)

def bs_high1(x):
    lower = [95, 110, 135, 150]
    upper = [90, 105, 140, 160]
    return interval_t2_mf(x, lower, upper)

def bs_high2(x):
    lower = [140, 160, 175, 190]
    upper = [135, 155, 180, 195]
    return interval_t2_mf(x, lower, upper)

def bs_high3(x):
    lower = [175, 190, 200, 200.01]
    upper = [170, 185, 200, 200.01]
    return interval_t2_mf(x, lower, upper)

# Define the fuzzy rules based on the input vital signs and the corresponding risk group output
def apply_rules(inputs):
    sbp_l3 = sbp_low3(inputs['SBP'])
    hr_l3 = hr_low3(inputs['HR'])
    spo2_l3 = spo2_low3(inputs['SPO2'])
    t_l3 = t_low3(inputs['T'])
    bs_l3 = bs_low3(inputs['BS'])

    # Example Rule 1: High risk if all are low+3
    rule1_lower = np.minimum.reduce([sbp_l3[0], hr_l3[0], spo2_l3[0], t_l3[0], bs_l3[0]])
    rule1_upper = np.minimum.reduce([sbp_l3[1], hr_l3[1], spo2_l3[1], t_l3[1], bs_l3[1]])

    sbp_nrm = sbp_normal(inputs['SBP'])
    hr_nrm = hr_normal(inputs['HR'])
    spo2_nrm = spo2_normal(inputs['SPO2'])
    t_nrm = t_normal(inputs['T'])
    bs_nrm = bs_normal(inputs['BS'])

    # Example Rule 2: Normal risk if all are normal
    rule2_lower = np.minimum.reduce([sbp_nrm[0], hr_nrm[0], spo2_nrm[0], t_nrm[0], bs_nrm[0]])
    rule2_upper = np.minimum.reduce([sbp_nrm[1], hr_nrm[1], spo2_nrm[1], t_nrm[1], bs_nrm[1]])

    # Combine results
    result_lower = np.maximum(rule1_lower * 14, rule2_lower * 0)
    result_upper = np.maximum(rule1_upper * 14, rule2_upper * 0)

    return result_lower.mean(), result_upper.mean()

# Example Input
inputs = {
    'SBP': 90,
    'HR': 80,
    'SPO2': 96,
    'T': 37.5,
    'BS': 100
}

# Apply the fuzzy logic rules
result = apply_rules(inputs)
print(f"Calculated Risk Group Interval: {result[0]} - {result[1]}")

# Visualize all membership functions
x_ranges = {
    'SBP': np.linspace(50, 200, 500),
    'HR': np.linspace(30, 150, 500),
    'SPO2': np.linspace(70, 100, 500),
    'T': np.linspace(35, 41, 500),
    'BS': np.linspace(50, 200, 500)
}

# Plot membership functions
for key, x in x_ranges.items():
    plt.figure(figsize=(10, 6))
    funcs = [f'{key.lower()}_low3', f'{key.lower()}_low2', f'{key.lower()}_low1', f'{key.lower()}_normal', f'{key.lower()}_high1', f'{key.lower()}_high2', f'{key.lower()}_high3']
    for func_name in funcs:
        func = globals().get(func_name)
        if func:
            lower, upper = func(x)
            plt.plot(x, lower, label=f'{func_name} lower', linestyle='--')
            plt.plot(x, upper, label=f'{func_name} upper', linestyle='-')
    plt.title(f'Interval Type-2 Membership Functions for {key}')
    plt.xlabel(key)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid()
    plt.show()
