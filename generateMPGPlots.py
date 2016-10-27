'''
generateMPGPlots.py
Requires Python 3.4+, numpy, and matplotlib
This file will display scatter plots comparing elements to MPG.
'''

from ProjectFunctions import best_fit_line
import numpy as np
import matplotlib.pyplot as plt

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

x_array = [cylinders, displacement, horsepower, weight, acceleration, model_year]

dataKey = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
lines_data = [False, False, False, False, False, False]

print('Best Fit Lines To MPG')
print('----------------------------------')

cylinder_m, cylinder_b = best_fit_line(cylinders, mpg)
lines_data[0] = (cylinder_m, cylinder_b)
print('cylinder: y =', cylinder_m, 'x +', cylinder_b)

displacement_m, displacement_b = best_fit_line(displacement, mpg)
lines_data[1] = (displacement_m, displacement_b)
print('displacement: y =', displacement_m, 'x +', displacement_b)

horsepower_m, horsepower_b = best_fit_line(horsepower, mpg)
lines_data[2] = (horsepower_m, horsepower_b)
print('horsepower: y =', horsepower_m, 'x +', horsepower_b)

weight_m, weight_b = best_fit_line(weight, mpg)
lines_data[3] = (weight_m, weight_b)
print('weight: y =', weight_m, 'x +', weight_b)

acceleration_m, acceleration_b = best_fit_line(acceleration, mpg)
lines_data[4] = (acceleration_m, acceleration_b)
print('acceleration: y =', acceleration_m, 'x +', acceleration_b)

model_year_m, model_year_b = best_fit_line(model_year, mpg)
lines_data[5] = (model_year_m, model_year_b)
print('model_year: y =', model_year_m, 'x +', model_year_b)


for i in range(6):
    try:
        label = 'MPG vs. ' + dataKey[i+1]
    except:
        break
    plt.scatter(np.array([car[i+1] for car in data], dtype=np.float64), mpg, label=label, color='k', s=25, marker="o")
    edges = [min(np.array([car[i+1] for car in data])), max(np.array([car[i+1] for car in data]))]
    plt.plot(edges, [((lines_data[i][0])*x)+(lines_data[i][1]) for x in edges])
    plt.xlabel(dataKey[i+1])
    plt.ylabel('Miles Per Gallon')
    plt.show()

print('done.')
