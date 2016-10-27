'''
printEquation.py
Requires Python 3.4+, numpy, and matplotlib
This file will print the equation arrived at for predicting mpg based on the
softmax_weight. This file accepts a adjustment weight as an arguement after the
file name.

Examples:
python predictMPG.py
=> Sets softmax_weight to 1
python predictMPG.py 12
=> Sets softmax_weight to 12
'''

from ProjectFunctions import best_fit_line, correlation, softmax, predict_mpg
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

try:
    softmax_weight = int(sys.argv[1])
except:
    softmax_weight = 1

mpgData = 'data/auto-mpg.data.txt'
dataKey = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']
with open(mpgData) as f:
    content = f.readlines()

data = [[float(x) for x in line.split()[0:7]] for line in content]
models = [re.findall(r'"([^"]*)"', line) for line in content]
mpg = np.array([car[0] for car in data], dtype=np.float64)
cylinders = np.array([car[1] for car in data], dtype=np.float64)
displacement = np.array([car[2] for car in data], dtype=np.float64)
horsepower = np.array([car[3] for car in data], dtype=np.float64)
weight = np.array([car[4] for car in data], dtype=np.float64)
acceleration = np.array([car[5] for car in data], dtype=np.float64)
model_year = np.array([car[6] for car in data], dtype=np.float64)

weight_to_mpg_m, weight_to_mpg_b = best_fit_line(weight, mpg)
cylinder_to_mpg_m, cylinder_to_mpg_b = best_fit_line(cylinders, mpg)
displacement_to_mpg_m, displacement_to_mpg_b = best_fit_line(displacement, mpg)
horsepower_to_mpg_m, horsepower_to_mpg_b = best_fit_line(horsepower, mpg)
acceleration_to_mpg_m, acceleration_to_mpg_b = best_fit_line(acceleration, mpg)
model_year_to_mpg_m, model_year_to_mpg_b = best_fit_line(model_year, mpg)

mpg_equations = [(cylinder_to_mpg_m, cylinder_to_mpg_b), (displacement_to_mpg_m, displacement_to_mpg_b), (horsepower_to_mpg_m, horsepower_to_mpg_b), (weight_to_mpg_m, weight_to_mpg_b), (acceleration_to_mpg_m, acceleration_to_mpg_b), (model_year_to_mpg_m, model_year_to_mpg_b)]

correlation_to_mpg_vector = [correlation(mpg, cylinders), correlation(mpg, displacement), correlation(mpg, horsepower), correlation(mpg, weight), correlation(mpg, acceleration), correlation(mpg, model_year)]
abs_correlation_vector = [abs(x) for x in correlation_to_mpg_vector]
test_value_vector = [num*(softmax_weight) for num in abs_correlation_vector]
weighting_values = softmax(test_value_vector)

b_vec = [weighting_values[i]*mpg_equations[i][1] for i in range(6) ]
m_vec = [weighting_values[i]*mpg_equations[i][0] for i in range(6) ]
f_intercept = sum(b_vec)
print('MPG =', m_vec[0],'cylinders + ',m_vec[1],'displacement + ', m_vec[2], ' horsepower + ', m_vec[3], 'weight + ', m_vec[4], 'acceleration + ', m_vec[5], 'model_year + ', f_intercept)
