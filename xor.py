# Xor.py
# Author: Hu Bowen
# Date: 14/9/22

# Jax implementation of a Neural Network from scratch
# XOR Test on network

# Libraries
import jax.numpy as jnp

from SimpleNeuralNet.Network import Network
from SimpleNeuralNet.Layers import FullyConnectedLayer, ActivationLayer
from SimpleNeuralNet.Activations import sigmoid, sigmoid_derivative
from SimpleNeuralNet.Loss_Functions import mse, mse_derivative

# Test data
inputs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = jnp.array([[0], [1], [1], [0]])

# Init network
net = Network()
net.add(FullyConnectedLayer(2, 1))
net.add(ActivationLayer(sigmoid, sigmoid_derivative))
net.add(FullyConnectedLayer(1, 1))
net.add(ActivationLayer(sigmoid, sigmoid_derivative))

# Train Network
net.set_loss_function(mse, mse_derivative)
net.train(inputs, expected_outputs, epochs=1000, learning_rate=0.1)

# Predict
print("\n========================\n")
prediction = net.predict(inputs)
print(prediction)