# Loss_functions.py
# Author: Hu Bowen
# Date: 14/9/22

# Jax implementation of a Neural Network from scratch
# Contains all loss functions and it's derivative

# Libraries
import jax
import jax.numpy as jnp

# Functions
@jax.jit
def mse(true_inputs, predicted_inputs):
    return jnp.mean(jnp.power(true_inputs - predicted_inputs, 2))

@jax.jit
def mse_derivative(true_inputs, predicted_inputs):
    return 2 * (predicted_inputs - true_inputs) / true_inputs.size

@jax.jit
def sse(true_inputs, predicted_inputs):
    return 0.5 * jnp.sum(jnp.power(true_inputs - predicted_inputs, 2))

@jax.jit
def sse_derivative(true_inputs, predicted_inputs):
    return predicted_inputs - true_inputs