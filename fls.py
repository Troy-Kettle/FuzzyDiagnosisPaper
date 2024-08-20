import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define the Antecedents (Inputs)
sbp = ctrl.Antecedent(np.arange(0, 200, 1), 'Systolic Blood Pressure')
hr = ctrl.Antecedent(np.arange(0, 140, 1), 'Heart Rate')
spo2 = ctrl.Antecedent(np.arange(80, 100, 1), 'SPO2')
temp = ctrl.Antecedent(np.arange(35, 40, 0.1), 'Temperature')
bs = ctrl.Antecedent(np.arange(60, 160, 1), 'Blood Sugar')

# Define the Consequent (Output)
risk = ctrl.Consequent(np.arange(0, 14, 0.5), 'Risk')

# Define Membership Functions using Trapezoidal Shapes
# Systolic Blood Pressure (SBP)
sbp['Low +3'] = fuzz.trapmf(sbp.universe, [0, 0, 70, 75])
sbp['Low +2'] = fuzz.trapmf(sbp.universe, [70, 75, 80, 85])
sbp['Low +1'] = fuzz.trapmf(sbp.universe, [80, 85, 95, 100])
sbp['Normal +0'] = fuzz.trapmf(sbp.universe, [95, 100, 180, 185])
sbp['High +2'] = fuzz.trapmf(sbp.universe, [180, 185, 200, 200])

# Heart Rate (HR)
hr['Low +2'] = fuzz.trapmf(hr.universe, [0, 0, 45, 50])
hr['Low +1'] = fuzz.trapmf(hr.universe, [45, 50, 55, 60])
hr['Normal +0'] = fuzz.trapmf(hr.universe, [53, 60, 95, 100])
hr['High +1'] = fuzz.trapmf(hr.universe, [95, 100, 105, 110])
hr['High +2'] = fuzz.trapmf(hr.universe, [105, 110, 125, 130])
hr['High +3'] = fuzz.trapmf(hr.universe, [125, 130, 140, 140])

# SPO2
spo2['Low +3'] = fuzz.trapmf(spo2.universe, [80, 80, 83, 85])
spo2['Low +2'] = fuzz.trapmf(spo2.universe, [83, 85, 87, 90])
spo2['Low +1'] = fuzz.trapmf(spo2.universe, [87, 90, 92, 95])
spo2['Normal +0'] = fuzz.trapmf(spo2.universe, [93, 95, 100, 100])

# Temperature
temp['Low +2'] = fuzz.trapmf(temp.universe, [35, 35, 36, 36.5])
temp['Normal +0'] = fuzz.trapmf(temp.universe, [36, 36.5, 38, 38.5])
temp['High +2'] = fuzz.trapmf(temp.universe, [38, 38.5, 40, 40])

# Blood Sugar
bs['Low +3'] = fuzz.trapmf(bs.universe, [60, 60, 63, 66])
bs['Low +2'] = fuzz.trapmf(bs.universe, [63, 66, 70, 72])
bs['Normal +0'] = fuzz.trapmf(bs.universe, [70, 72, 106, 110])
bs['High +2'] = fuzz.trapmf(bs.universe, [106, 110, 140, 150])
bs['High +3'] = fuzz.trapmf(bs.universe, [140, 150, 160, 160])

# Risk levels based on provided ranges
risk['NRM'] = fuzz.trapmf(risk.universe, [0, 0, 0.5, 1.5])
risk['LRG1'] = fuzz.trapmf(risk.universe, [0.5, 1.5, 1.5, 2.5])
risk['LRG2'] = fuzz.trapmf(risk.universe, [1.5, 2.5, 2.5, 3.5])
risk['LRG3'] = fuzz.trapmf(risk.universe, [2.5, 3.5, 3.5, 4.5])
risk['LRG4'] = fuzz.trapmf(risk.universe, [3.5, 4.5, 4.5, 5.5])
risk['HRG5'] = fuzz.trapmf(risk.universe, [4.5, 5.5, 5.5, 6.5])
risk['HRG6'] = fuzz.trapmf(risk.universe, [5.5, 6.5, 6.5, 7.5])
risk['HRG7'] = fuzz.trapmf(risk.universe, [6.5, 7.5, 7.5, 8.5])
risk['HRG8'] = fuzz.trapmf(risk.universe, [7.5, 8.5, 8.5, 9.5])
risk['HRG9'] = fuzz.trapmf(risk.universe, [8.5, 9.5, 9.5, 10.5])
risk['HRG10'] = fuzz.trapmf(risk.universe, [9.5, 10.5, 10.5, 11.5])
risk['HRG11'] = fuzz.trapmf(risk.universe, [10.5, 11.5, 11.5, 12.5])
risk['HRG12'] = fuzz.trapmf(risk.universe, [11.5, 12.5, 12.5, 13.5])
risk['HRG13'] = fuzz.trapmf(risk.universe, [12.5, 13.5, 13.5, 14])
risk['HRG14'] = fuzz.trapmf(risk.universe, [13.5, 14, 14, 14])

# Define the fuzzy rules (example rules)
rule1 = ctrl.Rule(sbp['Low +3'] | hr['Low +2'] | spo2['Low +3'] | temp['Low +2'] | bs['Low +3'], risk['HRG14'])
rule2 = ctrl.Rule(sbp['Normal +0'] & hr['Normal +0'] & spo2['Normal +0'] & temp['Normal +0'] & bs['Normal +0'], risk['NRM'])
rule3 = ctrl.Rule(sbp['High +2'] | hr['High +3'] | spo2['Low +2'] | temp['High +2'] | bs['High +3'], risk['HRG8'])

# Control system creation and simulation
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# Example: Input values
risk_sim.input['Systolic Blood Pressure'] = 110
risk_sim.input['Heart Rate'] = 70
risk_sim.input['SPO2'] = 96
risk_sim.input['Temperature'] = 37.5
risk_sim.input['Blood Sugar'] = 90

# Compute the risk
risk_sim.compute()

# Print the output risk level
print(f"Calculated Risk Level: {risk_sim.output['Risk']}")

# Function to plot trapezoidal membership functions
def plot_mf(ax, universe, mfs, title):
    for label, mf in mfs.items():
        y = fuzz.trapmf(universe, mf)
        ax.plot(universe, y, label=label)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('Membership')

# Create the plot
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('MEWS-based Categorization Membership Functions', fontsize=16)

# Systolic Blood Pressure (SBP)
plot_mf(axs[0, 0], sbp.universe, {
    'Low +3': [0, 0, 70, 75],
    'Low +2': [70, 75, 80, 85],
    'Low +1': [80, 85, 95, 100],
    'Normal +0': [95, 100, 180, 185],
    'High +2': [180, 185, 200, 200]
}, 'Systolic Blood Pressure (SBP)')

# Heart Rate (HR)
plot_mf(axs[0, 1], hr.universe, {
    'Low +2': [0, 0, 45, 50],
    'Low +1': [45, 50, 55, 60],
    'Normal +0': [53, 60, 95, 100],
    'High +1': [95, 100, 105, 110],
    'High +2': [105, 110, 125, 130],
    'High +3': [125, 130, 140, 140]
}, 'Heart Rate (HR)')

# SPO2
plot_mf(axs[1, 0], spo2.universe, {
    'Low +3': [80, 80, 83, 85],
    'Low +2': [83, 85, 87, 90],
    'Low +1': [87, 90, 92, 95],
    'Normal +0': [93, 95, 100, 100]
}, 'SPO2')

# Temperature
plot_mf(axs[1, 1], temp.universe, {
    'Low +2': [35, 35, 36, 36.5],
    'Normal +0': [36, 36.5, 38, 38.5],
    'High +2': [38, 38.5, 40, 40]
}, 'Temperature')

# Blood Sugar
plot_mf(axs[2, 0], bs.universe, {
    'Low +3': [60, 60, 63, 66],
    'Low +2': [63, 66, 70, 72],
    'Normal +0': [70, 72, 106, 110],
    'High +2': [106, 110, 140, 150],
    'High +3': [140, 150, 160, 160]
}, 'Blood Sugar')

# Risk
plot_mf(axs[2, 1], risk.universe, {
    'NRM': [0, 0, 0.5, 1.5],
    'LRG1': [0.5, 1.5, 1.5, 2.5],
    'LRG2': [1.5, 2.5, 2.5, 3.5],
    'LRG3': [2.5, 3.5, 3.5, 4.5],
    'LRG4': [3.5, 4.5, 4.5, 5.5],
    'HRG5': [4.5, 5.5, 5.5, 6.5],
    'HRG6': [5.5, 6.5, 6.5, 7.5],
    'HRG7': [6.5, 7.5, 7.5, 8.5],
    'HRG8': [7.5, 8.5, 8.5, 9.5],
    'HRG9': [8.5, 9.5, 9.5, 10.5],
    'HRG10': [9.5, 10.5, 10.5, 11.5],
    'HRG11': [10.5, 11.5, 11.5, 12.5],
    'HRG12': [11.5, 12.5, 12.5, 13.5],
    'HRG13': [12.5, 13.5, 13.5, 14],
    'HRG14': [13.5, 14, 14, 14]
}, 'Risk')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
