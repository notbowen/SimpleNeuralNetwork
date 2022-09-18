# Layers.py
# Author: Hu Bowen
# Date: 14/9/22

# Jax implementation of a Neural Network from scratch
# Layers of the neural network

# Libraries
import jax
import jax.numpy as jnp

import time

# Init random key
key = jax.random.PRNGKey(int(time.time()))

# Base layer class
class Layer:
    def __init__(self) -> None:
        self.inputs = None
        self.outputs = None

    def forward_propagate(self, inputs):
        raise NotImplementedError

    def backward_propagate(self, output_errors, learning_rate):
        raise NotImplementedError

# Fully connected layer
# All neurons are connected to other neurons
class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = jax.random.uniform(key, (input_size, output_size)) / jnp.sqrt(input_size + output_size)
        self.bias = jax.random.uniform(key, (1, output_size)) / jnp.sqrt(input_size + output_size)
        
    # Forward propagation
    def forward_propagate(self, inputs):
        self.inputs = inputs
        self.outputs = jnp.dot(self.inputs, self.weights) + self.bias  # output = (W1 A1 + W2 A2 ... + Wn An) + bias
        #self.outputs=jnp.add(jnp.multiply(self.weights*self.inputs),self.bias) where all inputs are jnp.arrays
        return self.outputs

    def backward_propagate(self, output_errors, learning_rate):
        # Calculate change
        input_error = jnp.dot(output_errors, self.weights.T)
        weights_error = jnp.dot(self.inputs.T, output_errors)

        # Gradient descent
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_errors

        return input_error

# Activation layer
# Calculates a layer's activation
class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation

    # Returns activation value
    def forward_propagate(self, input_data):
        self.inputs = input_data
        self.outputs = self.activation(self.inputs)

        return self.outputs

    # Returns change in input for a given output
    def backward_propagate(self, output_error, learning_rate):
        # return self.activation_derivative(self.inputs) * output_error
        return jax.vjp(self.activation, self.inputs)[0] * output_error
