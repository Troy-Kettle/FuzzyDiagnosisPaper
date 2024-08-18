import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create the universe of discourse for each vital sign
SBP = ctrl.Antecedent(np.arange(50, 201, 1), 'SBP')  # Systolic Blood Pressure
HR = ctrl.Antecedent(np.arange(30, 150, 1), 'HR')    # Heart Rate
SPO2 = ctrl.Antecedent(np.arange(70, 100, 1), 'SPO2') # Blood Oxygen Level
T = ctrl.Antecedent(np.arange(35, 41, 0.1), 'T')      # Temperature
BS = ctrl.Antecedent(np.arange(50, 200, 1), 'BS')     # Blood Sugar

# Create the universe of discourse for the output (Risk Group)
RG = ctrl.Consequent(np.arange(0, 15, 0.1), 'RG')

# Define membership functions for RG (Risk Group)
RG['NRM'] = fuzz.trimf(RG.universe, [0, 0.25, 0.5])
RG['LRG1'] = fuzz.trimf(RG.universe, [0.5, 1, 1.5])
RG['LRG2'] = fuzz.trimf(RG.universe, [1.5, 2, 2.5])
RG['LRG3'] = fuzz.trimf(RG.universe, [2.5, 3, 3.5])
RG['LRG4'] = fuzz.trimf(RG.universe, [3.5, 4, 4.5])
RG['HRG5'] = fuzz.trimf(RG.universe, [4.5, 5, 5.5])
RG['HRG6'] = fuzz.trimf(RG.universe, [5.5, 6, 6.5])
RG['HRG7'] = fuzz.trimf(RG.universe, [6.5, 7, 7.5])
RG['HRG8'] = fuzz.trimf(RG.universe, [7.5, 8, 8.5])
RG['HRG9'] = fuzz.trimf(RG.universe, [8.5, 9, 9.5])
RG['HRG10'] = fuzz.trimf(RG.universe, [9.5, 10, 10.5])
RG['HRG11'] = fuzz.trimf(RG.universe, [10.5, 11, 11.5])
RG['HRG12'] = fuzz.trimf(RG.universe, [11.5, 12, 12.5])
RG['HRG13'] = fuzz.trimf(RG.universe, [12.5, 13, 13.5])
RG['HRG14'] = fuzz.trimf(RG.universe, [13.5, 13.75, 14])

# Define membership functions for Systolic Blood Pressure (SBP)
SBP['Low+3'] = fuzz.trimf(SBP.universe, [50, 50, 75])
SBP['Low+2'] = fuzz.trimf(SBP.universe, [70, 75, 85])
SBP['Low+1'] = fuzz.trimf(SBP.universe, [80, 90, 100])
SBP['Normal'] = fuzz.trimf(SBP.universe, [95, 125, 199])
SBP['High+1'] = fuzz.trimf(SBP.universe, [185, 200, 200])
SBP['High+2'] = fuzz.trimf(SBP.universe, [185, 200, 200])
SBP['High+3'] = fuzz.trimf(SBP.universe, [185, 200, 200])

# Define membership functions for Heart Rate (HR)
HR['Low+3'] = fuzz.trimf(HR.universe, [30, 30, 50])
HR['Low+2'] = fuzz.trimf(HR.universe, [45, 50, 60])
HR['Low+1'] = fuzz.trimf(HR.universe, [53, 60, 100])
HR['Normal'] = fuzz.trimf(HR.universe, [53, 70, 100])
HR['High+1'] = fuzz.trimf(HR.universe, [95, 110, 125])
HR['High+2'] = fuzz.trimf(HR.universe, [105, 115, 130])
HR['High+3'] = fuzz.trimf(HR.universe, [125, 150, 150])

# Define membership functions for SPO2
SPO2['Low+3'] = fuzz.trimf(SPO2.universe, [70, 70, 85])
SPO2['Low+2'] = fuzz.trimf(SPO2.universe, [83, 85, 90])
SPO2['Low+1'] = fuzz.trimf(SPO2.universe, [87, 90, 95])
SPO2['Normal'] = fuzz.trimf(SPO2.universe, [93, 95, 100])

# Define membership functions for Temperature (T)
T['Low+3'] = fuzz.trimf(T.universe, [35, 35, 36.5])
T['Low+2'] = fuzz.trimf(T.universe, [36.5, 36.5, 36.5])
T['Normal'] = fuzz.trimf(T.universe, [36, 38.5, 38.5])
T['High+1'] = fuzz.trimf(T.universe, [38, 39, 40])
T['High+2'] = fuzz.trimf(T.universe, [38, 40, 41])

# Define membership functions for Blood Sugar (BS)
BS['Low+3'] = fuzz.trimf(BS.universe, [50, 50, 66])
BS['Low+2'] = fuzz.trimf(BS.universe, [63, 66, 72])
BS['Normal'] = fuzz.trimf(BS.universe, [70, 100, 110])
BS['High+1'] = fuzz.trimf(BS.universe, [106, 120, 150])
BS['High+2'] = fuzz.trimf(BS.universe, [106, 120, 150])
BS['High+3'] = fuzz.trimf(BS.universe, [140, 180, 200])

# Define the fuzzy rules based on the input vital signs and the corresponding risk group output
rules = [
    ctrl.Rule(SBP['Low+3'] | HR['Low+3'] | SPO2['Low+3'] | T['Low+3'] | BS['Low+3'], RG['HRG14']),
    ctrl.Rule(SBP['Low+2'] | HR['Low+2'] | SPO2['Low+2'] | T['Low+2'] | BS['Low+2'], RG['HRG13']),
    ctrl.Rule(SBP['Low+1'] | HR['Low+1'] | SPO2['Low+1'] | T['Normal'] | BS['Normal'], RG['LRG4']),
    ctrl.Rule(SBP['Normal'] & HR['Normal'] & SPO2['Normal'] & T['Normal'] & BS['Normal'], RG['NRM']),
    ctrl.Rule(SBP['High+1'] | HR['High+1'] | T['High+1'] | BS['High+1'], RG['LRG3']),
    ctrl.Rule(SBP['High+2'] | HR['High+2'] | T['High+2'] | BS['High+2'], RG['HRG6']),
    ctrl.Rule(SBP['High+3'] | HR['High+3'], RG['HRG7']),
]

# Create the control system and simulation
rg_ctrl = ctrl.ControlSystem(rules)
rg_simulation = ctrl.ControlSystemSimulation(rg_ctrl)

# Example Input
rg_simulation.input['SBP'] = 90
rg_simulation.input['HR'] = 80
rg_simulation.input['SPO2'] = 96
rg_simulation.input['T'] = 37.5
rg_simulation.input['BS'] = 100

# Compute the result
rg_simulation.compute()

# Output the risk group value
print(f"Calculated Risk Group: {rg_simulation.output['RG']}")

# View membership functions for each input
SBP.view()
HR.view()
SPO2.view()
T.view()
BS.view()

# View membership function for the Risk Group output with the calculated value
RG.view(sim=rg_simulation)

plt.show()
