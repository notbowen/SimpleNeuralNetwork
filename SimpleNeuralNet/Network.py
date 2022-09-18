# Network.py
# Author: Hu Bowen
# Date: 14/9/22

# Jax implementation of a Neural Network from scratch
# Main network script, forward and backward propagates layers of neurons

# TODO: Implement saving of weights and biases

# Libraries
import jax

# Network class
class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    # Train network
    def train(self, inputs, expected_outputs, epochs, learning_rate):
        samples = len(inputs)

        # Training
        for epoch in range(epochs):
            error_display = 0
            for sample in range(samples):
                output = inputs[sample]

                for layer in self.layers:
                    output = layer.forward_propagate(output)  # Pass inputs through network
                
                # Output loss
                error_display += self.loss_function(expected_outputs[sample], output)

                # Backward propagate
                error = jax.vjp(self.loss_function, expected_outputs[sample], output)[0]
                # error = self.loss_function_derivative(expected_outputs[sample], output)
                for layer in reversed(self.layers): 
                    error = layer.backward_propagate(error, learning_rate)

            # Display avg error and epoch
            error_display /= samples
            print(f"Epoch: {epoch+1}/{epochs}, Error: {error_display}")

    # Predict 
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for sample in range(samples):
            # Forward propagate thru network
            output = input_data[sample]
            for layer in self.layers:
                output = layer.forward_propagate(output)
            result.append(output)

        return result