'''
ProjectFunctions.py
requires python 3.4+ and numpy
This file contains functions used in the other files
'''

from statistics import mean
import numpy as np

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)

def mean_deviation(x):
    mean_value = mean(x)
    return [y - mean_value for y in x]

def covariance(x, y):
    n = len(x)
    return np.dot(mean_deviation(x), mean_deviation(y)) / (n - 1)

def correlation(x, y):
    stdev_x = np.std(x)
    stdev_y = np.std(y)
    if stdev_x > 0 and stdev_y > 0:
        return (covariance(x, y) / stdev_x) / stdev_y
    else:
        return 0

def best_fit_line(xs,ys):
    m = (((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m,b

def find_y(m, b, x):
    return (m*x)+b

def predict_mpg(equations, weights, values):
    predicted_mpg = 0
    for i in range(6):
        predicted_mpg = predicted_mpg + weights[i]*((equations[i][0]*values[i+1])+equations[i][1])
    return predicted_mpg

def find_sum_errors_squared(data, mpg_equations, weighting_values):
    result = 0
    for car in data:
        error = car[0] - predict_mpg(mpg_equations, weighting_values, car)
        result = result + (error**2)
    return result

def find_range_adjustment_results(data, mpg_equations, abs_corr_vector):
    results = []
    for i in range(-100,100):
        test_value_vector = [num*(i) for num in abs_corr_vector] if i >= 0 else [num/(-i) for num in abs_corr_vector]
        weights_vector = softmax(test_value_vector)
        errors_squared = find_sum_errors_squared(data, mpg_equations, weights_vector)
        print('(',i,',', errors_squared,')')
        results.append(errors_squared)
    return results

def find_sum_abs_errors(data, mpg_equations, weighting_values):
    result = 0
    for car in data:
        error = car[0] - predict_mpg(mpg_equations, weighting_values, car)
        result = result + abs(error)
    return result

def find_range_adjustment_results_abs(data, mpg_equations, abs_corr_vector):
    results = []
    for i in range(-100, 100):
        test_value_vector = [num*(i) for num in abs_corr_vector] if i >= 0 else [num/(-i) for num in abs_corr_vector]
        weights_vector = softmax(test_value_vector)
        errors_squared = find_sum_abs_errors(data, mpg_equations, weights_vector)
        print('(',i,',',errors_squared,')')
        results.append(errors_squared)
    return results
