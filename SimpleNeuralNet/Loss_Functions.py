# Loss_functions.py
# Author: Hu Bowen
# Date: 14/9/22

# Implementation of a Neural Network from scratch
# Contains all loss functions and it's derivative

# Libraries
import numpy as np

# Functions
def mse(true_inputs, predicted_inputs):
    return np.mean(np.power(true_inputs - predicted_inputs, 2))

def mse_derivative(true_inputs, predicted_inputs):
    return 2 * (predicted_inputs - true_inputs) / true_inputs.size

def sse(true_inputs, predicted_inputs):
    return 0.5 * np.sum(np.power(true_inputs - predicted_inputs, 2))

def sse_derivative(true_inputs, predicted_inputs):
    return predicted_inputs - true_inputs