# Xor.py
# Author: Hu Bowen
# Date: 14/9/22

# Implementation of a Neural Network from scratch
# XOR Test on network

# Libraries
import jax.numpy as jnp

from SimpleNeuralNet.Network import Network
from SimpleNeuralNet.Layers import FullyConnectedLayer, ActivationLayer
from SimpleNeuralNet.Activations import relu
from SimpleNeuralNet.Loss_Functions import sse

# Test data
inputs = jnp.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
expected_outputs = jnp.array([[0], [1], [1], [0]])

# Init network
net = Network()
net.add(FullyConnectedLayer(2, 3))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(3, 1))
net.add(ActivationLayer(relu))

# Train Network
net.set_loss_function(sse)
net.train(inputs, expected_outputs, epochs=1000, learning_rate=0.1)

# Predict
print("\n========================\n")
prediction = net.predict(inputs)
print("         Expected Predicted (rounded) Predicted")
for i in range(len(inputs)):
    print(f"{inputs[i][0][0]} XOR {inputs[i][0][1]}: {expected_outputs[i][0]}        {round(prediction[i][0][0])}         (rounded) {prediction[i][0][0]}")