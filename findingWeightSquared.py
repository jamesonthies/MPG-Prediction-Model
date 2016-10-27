'''
findingWeightSquared.py
Requires Python 3.4+, numpy, and matplotlib
This file displays a plot showing the sum of errors squared for possible
weights.
'''

from ProjectFunctions import correlation, best_fit_line, find_range_adjustment_results
import matplotlib.pyplot as plt
import numpy as np

mpgData = 'data/auto-mpg.data.txt'
dataKey = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']
with open(mpgData) as f:
    content = f.readlines()

data = [[float(x) for x in line.split()[0:8]] for line in content]
mpg = np.array([car[0] for car in data], dtype=np.float64)
cylinders = np.array([car[1] for car in data], dtype=np.float64)
displacement = np.array([car[2] for car in data], dtype=np.float64)
horsepower = np.array([car[3] for car in data], dtype=np.float64)
weight = np.array([car[4] for car in data], dtype=np.float64)
acceleration = np.array([car[5] for car in data], dtype=np.float64)
model_year = np.array([car[6] for car in data], dtype=np.float64)
origin = np.array([car[7] for car in data], dtype=np.float64)

corr_vector_to_mpg = [correlation(mpg, cylinders), correlation(mpg, displacement), correlation(mpg, horsepower), correlation(mpg, weight), correlation(mpg, acceleration), correlation(mpg, model_year)]
abs_corr_vector = [abs(x) for x in corr_vector_to_mpg]

weight_to_mpg_m, weight_to_mpg_b = best_fit_line(weight, mpg)
cylinder_to_mpg_m, cylinder_to_mpg_b = best_fit_line(cylinders, mpg)
displacement_to_mpg_m, displacement_to_mpg_b = best_fit_line(displacement, mpg)
horsepower_to_mpg_m, horsepower_to_mpg_b = best_fit_line(horsepower, mpg)
acceleration_to_mpg_m, acceleration_to_mpg_b = best_fit_line(acceleration, mpg)
model_year_to_mpg_m, model_year_to_mpg_b = best_fit_line(model_year, mpg)
mpg_equations = [(cylinder_to_mpg_m, cylinder_to_mpg_b), (displacement_to_mpg_m, displacement_to_mpg_b), (horsepower_to_mpg_m, horsepower_to_mpg_b), (weight_to_mpg_m, weight_to_mpg_b), (acceleration_to_mpg_m, acceleration_to_mpg_b), (model_year_to_mpg_m, model_year_to_mpg_b)]

possibilities = find_range_adjustment_results(data, mpg_equations, abs_corr_vector)

plt.scatter([_ for _ in range(-100,100)], possibilities, label='weight adjustment values', color='k', s=25, marker="o")
plt.xlabel('Adjustment Weight')
plt.ylabel('Sum of Errors Squared')
plt.show()

print('done.')
