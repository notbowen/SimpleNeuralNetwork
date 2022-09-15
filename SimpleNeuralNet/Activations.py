# Activations.py
# Author: Hu Bowen
# Date: 14/9/22

# Jax implementation of a Neural Network from scratch
# Contains all activation functions and it's derivative

# Libraries
import jax
import jax.numpy as jnp

# Functions
@jax.jit
def tanh(x):
    return jax.nn.tanh(x)

@jax.jit
def tanh_derivative(x):
    return 1 - jax.nn.tanh(x) ** 2

@jax.jit
def sigmoid(x):
    return jax.nn.sigmoid(x)

@jax.jit
def sigmoid_derivative(x):
    return jnp.exp(-x) / (1 + jnp.exp(-x)) ** 2

@jax.jit
def relu(x):
    return jnp.maximum(x, 0)

@jax.jit
def relu_derivative(x):
    return jnp.array(x >= 0).astype('int')