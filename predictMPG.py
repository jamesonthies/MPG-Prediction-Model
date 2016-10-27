'''
predictMPG.py
Requires Python 3.4+, numpy, and matplotlib
This file will print predicted mpg for every car in the data set. It also
accepts an arguement after the file name to apply as the adjustment weight,
the default being one. It will display a graph comparing the projections with
known values.

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



#Uncomment following lines and add a text file to the 'file_name' variable to log results
#file_name = 'FILE_LOCATION_HERE'
#log_file = open(file_name, 'w')
#log_file.write('(model, predicted_mpg, actual_mpg)\n')

#i = 0
predicted_mpgs_list = []
actual_mpgs_list = []
for car in data:
    actual_mpg = car[0]
    car[0] = False
    current_car = car
    predicted_mpg = predict_mpg(mpg_equations, weighting_values, current_car)
    predicted_mpgs_list.append(predicted_mpg)
    actual_mpgs_list.append(actual_mpg)
    #log_file.write('(')
    #log_file.write(models[i][0])
    #log_file.write(', ')
    #log_file.write(str(predicted_mpg))
    #log_file.write(', ')
    #log_file.write(str(actual_mpg))
    #log_file.write(')\n')
    #i = i + 1

np_actual_mpgs= np.array([x for x in actual_mpgs_list], dtype=np.float64)
np_predicted_mpgs= np.array([x for x in predicted_mpgs_list], dtype=np.float64)
best_fit_m, best_fit_b = best_fit_line(np_actual_mpgs, np_predicted_mpgs)
regression_min = best_fit_b
regression_max = (50*best_fit_m)+best_fit_b

plt.scatter(np_actual_mpgs,np_predicted_mpgs, label='predicted and actual mpg plot', color='k', s=25, marker="o")
plt.xlabel('Actual MPG')
plt.axis([0,50,0,50])
plt.ylabel('Predicted MPG')
plt.plot([0,50],[0,50], color='g')
plt.plot([0,50],[regression_min, regression_max],color='b')
plt.show()

print('done.')
