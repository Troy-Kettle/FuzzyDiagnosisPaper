import numpy as np
import matplotlib.pyplot as plt

# Define trapezoidal membership function
def trapezoidal_mf(x, a, b, c, d):
    return np.maximum(0, np.minimum((x-a)/(b-a), np.minimum(1, (d-x)/(d-c))))

# Define interval type-2 membership function
def interval_t2_mf(x, lower, upper):
    lower_mf = trapezoidal_mf(x, *lower)
    upper_mf = trapezoidal_mf(x, *upper)
    return lower_mf, upper_mf

# Define membership functions for each input
def sbp_low3(x):
    lower = [50, 50, 75, 75]
    upper = [50, 55, 70, 75]
    return interval_t2_mf(x, lower, upper)

def sbp_low2(x):
    lower = [70, 75, 85, 85]
    upper = [70, 80, 85, 90]
    return interval_t2_mf(x, lower, upper)

def sbp_low1(x):
    lower = [80, 90, 100, 100]
    upper = [80, 95, 105, 100]
    return interval_t2_mf(x, lower, upper)

def sbp_normal(x):
    lower = [95, 125, 199, 199]
    upper = [95, 130, 200, 200]
    return interval_t2_mf(x, lower, upper)

def sbp_high1(x):
    lower = [185, 200, 200, 200]
    upper = [185, 200, 200, 200]
    return interval_t2_mf(x, lower, upper)

def sbp_high2(x):
    lower = [185, 200, 200, 200]
    upper = [185, 200, 200, 200]
    return interval_t2_mf(x, lower, upper)

def sbp_high3(x):
    lower = [185, 200, 200, 200]
    upper = [185, 200, 200, 200]
    return interval_t2_mf(x, lower, upper)

# Repeat similar functions for HR, SPO2, T, and BS

# Define the function to apply fuzzy rules
def apply_rules(inputs):
    x_sbp = np.array(inputs['SBP'])
    x_hr = np.array(inputs['HR'])
    x_spo2 = np.array(inputs['SPO2'])
    x_t = np.array(inputs['T'])
    x_bs = np.array(inputs['BS'])

    # Get membership values for all inputs
    sbp_l3 = sbp_low3(x_sbp)
    sbp_l2 = sbp_low2(x_sbp)
    sbp_l1 = sbp_low1(x_sbp)
    sbp_nrm = sbp_normal(x_sbp)
    sbp_h1 = sbp_high1(x_sbp)
    sbp_h2 = sbp_high2(x_sbp)
    sbp_h3 = sbp_high3(x_sbp)

    # Repeat for other inputs

    # Calculate the rule results
    rule1_lower = np.minimum.reduce([sbp_l3[0], hr_l3[0], spo2_l3[0], t_l3[0], bs_l3[0]])
    rule1_upper = np.minimum.reduce([sbp_l3[1], hr_l3[1], spo2_l3[1], t_l3[1], bs_l3[1]])

    rule2_lower = np.minimum.reduce([sbp_l2[0], hr_l2[0], spo2_l2[0], t_l2[0], bs_l2[0]])
    rule2_upper = np.minimum.reduce([sbp_l2[1], hr_l2[1], spo2_l2[1], t_l2[1], bs_l2[1]])

    rule3_lower = np.minimum.reduce([sbp_l1[0], hr_l1[0], spo2_l1[0], t_nrm[0], bs_nrm[0]])
    rule3_upper = np.minimum.reduce([sbp_l1[1], hr_l1[1], spo2_l1[1], t_nrm[1], bs_nrm[1]])

    rule4_lower = np.minimum.reduce([sbp_nrm[0], hr_nrm[0], spo2_nrm[0], t_nrm[0], bs_nrm[0]])
    rule4_upper = np.minimum.reduce([sbp_nrm[1], hr_nrm[1], spo2_nrm[1], t_nrm[1], bs_nrm[1]])

    rule5_lower = np.minimum.reduce([sbp_h1[0], hr_h1[0], t_h1[0], bs_h1[0]])
    rule5_upper = np.minimum.reduce([sbp_h1[1], hr_h1[1], t_h1[1], bs_h1[1]])

    rule6_lower = np.minimum.reduce([sbp_h2[0], hr_h2[0], t_h2[0], bs_h2[0]])
    rule6_upper = np.minimum.reduce([sbp_h2[1], hr_h2[1], t_h2[1], bs_h2[1]])

    rule7_lower = np.minimum.reduce([sbp_h3[0], hr_h3[0]])
    rule7_upper = np.minimum.reduce([sbp_h3[1], hr_h3[1]])

    # Aggregating the rules
    aggregated_lower = np.maximum.reduce([rule1_lower, rule2_lower, rule3_lower, rule4_lower, rule5_lower, rule6_lower, rule7_lower])
    aggregated_upper = np.maximum.reduce([rule1_upper, rule2_upper, rule3_upper, rule4_upper, rule5_upper, rule6_upper, rule7_upper])

    # Calculating the output
    result_lower = np.sum(aggregated_lower)
    result_upper = np.sum(aggregated_upper)

    return (result_lower + result_upper) / 2

# Example Input for IT2 FLS
inputs = {
    'SBP': 90,
    'HR': 80,
    'SPO2': 96,
    'T': 37.5,
    'BS': 100
}

# Compute the IT2 FLS result
risk_group_it2 = apply_rules(inputs)
print(f"Calculated Risk Group (IT2 FLS): {risk_group_it2}")

# Plot Membership Functions
x_sbp = np.arange(50, 201, 1)
x_hr = np.arange(30, 150, 1)
x_spo2 = np.arange(70, 100, 1)
x_t = np.arange(35, 41, 0.1)
x_bs = np.arange(50, 200, 1)

# SBP Membership Functions
plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.plot(x_sbp, sbp_low3(x_sbp)[0], label='SBP Low+3 Lower')
plt.plot(x_sbp, sbp_low3(x_sbp)[1], label='SBP Low+3 Upper')
plt.plot(x_sbp, sbp_low2(x_sbp)[0], label='SBP Low+2 Lower')
plt.plot(x_sbp, sbp_low2(x_sbp)[1], label='SBP Low+2 Upper')
plt.plot(x_sbp, sbp_low1(x_sbp)[0], label='SBP Low+1 Lower')
plt.plot(x_sbp, sbp_low1(x_sbp)[1], label='SBP Low+1 Upper')
plt.plot(x_sbp, sbp_normal(x_sbp)[0], label='SBP Normal Lower')
plt.plot(x_sbp, sbp_normal(x_sbp)[1], label='SBP Normal Upper')
plt.plot(x_sbp, sbp_high1(x_sbp)[0], label='SBP High+1 Lower')
plt.plot(x_sbp, sbp_high1(x_sbp)[1], label='SBP High+1 Upper')
plt.plot(x_sbp, sbp_high2(x_sbp)[0], label='SBP High+2 Lower')
plt.plot(x_sbp, sbp_high2(x_sbp)[1], label='SBP High+2 Upper')
plt.plot(x_sbp, sbp_high3(x_sbp)[0], label='SBP High+3 Lower')
plt.plot(x_sbp, sbp_high3(x_sbp)[1], label='SBP High+3 Upper')
plt.title('SBP Membership Functions')
plt.xlabel('SBP')
plt.ylabel('Membership Degree')
plt.legend()

# HR Membership Functions
plt.subplot(3, 2, 2)
plt.plot(x_hr, hr_low3(x_hr)[0], label='HR Low+3 Lower')
plt.plot(x_hr, hr_low3(x_hr)[1], label='HR Low+3 Upper')
plt.plot(x_hr, hr_low2(x_hr)[0], label='HR Low+2 Lower')
plt.plot(x_hr, hr_low2(x_hr)[1], label='HR Low+2 Upper')
plt.plot(x_hr, hr_low1(x_hr)[0], label='HR Low+1 Lower')
plt.plot(x_hr, hr_low1(x_hr)[1], label='HR Low+1 Upper')
plt.plot(x_hr, hr_normal(x_hr)[0], label='HR Normal Lower')
plt.plot(x_hr, hr_normal(x_hr)[1], label='HR Normal Upper')
plt.plot(x_hr, hr_high1(x_hr)[0], label='HR High+1 Lower')
plt.plot(x_hr, hr_high1(x_hr)[1], label='HR High+1 Upper')
plt.plot(x_hr, hr_high2(x_hr)[0], label='HR High+2 Lower')
plt.plot(x_hr, hr_high2(x_hr)[1], label='HR High+2 Upper')
plt.plot(x_hr, hr_high3(x_hr)[0], label='HR High+3 Lower')
plt.plot(x_hr, hr_high3(x_hr)[1], label='HR High+3 Upper')
plt.title('HR Membership Functions')
plt.xlabel('Heart Rate')
plt.ylabel('Membership Degree')
plt.legend()

# SPO2 Membership Functions
plt.subplot(3, 2, 3)
plt.plot(x_spo2, spo2_low3(x_spo2)[0], label='SPO2 Low+3 Lower')
plt.plot(x_spo2, spo2_low3(x_spo2)[1], label='SPO2 Low+3 Upper')
plt.plot(x_spo2, spo2_low2(x_spo2)[0], label='SPO2 Low+2 Lower')
plt.plot(x_spo2, spo2_low2(x_spo2)[1], label='SPO2 Low+2 Upper')
plt.plot(x_spo2, spo2_low1(x_spo2)[0], label='SPO2 Low+1 Lower')
plt.plot(x_spo2, spo2_low1(x_spo2)[1], label='SPO2 Low+1 Upper')
plt.plot(x_spo2, spo2_normal(x_spo2)[0], label='SPO2 Normal Lower')
plt.plot(x_spo2, spo2_normal(x_spo2)[1], label='SPO2 Normal Upper')
plt.title('SPO2 Membership Functions')
plt.xlabel('SPO2')
plt.ylabel('Membership Degree')
plt.legend()

# Temperature Membership Functions
plt.subplot(3, 2, 4)
plt.plot(x_t, t_low3(x_t)[0], label='T Low+3 Lower')
plt.plot(x_t, t_low3(x_t)[1], label='T Low+3 Upper')
plt.plot(x_t, t_low2(x_t)[0], label='T Low+2 Lower')
plt.plot(x_t, t_low2(x_t)[1], label='T Low+2 Upper')
plt.plot(x_t, t_normal(x_t)[0], label='T Normal Lower')
plt.plot(x_t, t_normal(x_t)[1], label='T Normal Upper')
plt.plot(x_t, t_high1(x_t)[0], label='T High+1 Lower')
plt.plot(x_t, t_high1(x_t)[1], label='T High+1 Upper')
plt.plot(x_t, t_high2(x_t)[0], label='T High+2 Lower')
plt.plot(x_t, t_high2(x_t)[1], label='T High+2 Upper')
plt.title('Temperature Membership Functions')
plt.xlabel('Temperature')
plt.ylabel('Membership Degree')
plt.legend()

# Blood Sugar Membership Functions
plt.subplot(3, 2, 5)
plt.plot(x_bs, bs_low3(x_bs)[0], label='BS Low+3 Lower')
plt.plot(x_bs, bs_low3(x_bs)[1], label='BS Low+3 Upper')
plt.plot(x_bs, bs_low2(x_bs)[0], label='BS Low+2 Lower')
plt.plot(x_bs, bs_low2(x_bs)[1], label='BS Low+2 Upper')
plt.plot(x_bs, bs_normal(x_bs)[0], label='BS Normal Lower')
plt.plot(x_bs, bs_normal(x_bs)[1], label='BS Normal Upper')
plt.plot(x_bs, bs_high1(x_bs)[0], label='BS High+1 Lower')
plt.plot(x_bs, bs_high1(x_bs)[1], label='BS High+1 Upper')
plt.plot(x_bs, bs_high2(x_bs)[0], label='BS High+2 Lower')
plt.plot(x_bs, bs_high2(x_bs)[1], label='BS High+2 Upper')
plt.plot(x_bs, bs_high3(x_bs)[0], label='BS High+3 Lower')
plt.plot(x_bs, bs_high3(x_bs)[1], label='BS High+3 Upper')
plt.title('Blood Sugar Membership Functions')
plt.xlabel('Blood Sugar')
plt.ylabel('Membership Degree')
plt.legend()

plt.tight_layout()
plt.show()
