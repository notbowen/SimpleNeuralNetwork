# Activations.py
# Author: Hu Bowen
# Date: 14/9/22

# Implementation of a Neural Network from scratch
# Contains all activation functions and it's derivative

# Libraries
import numpy as np

# Functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.array(x >= 0).astype('int')