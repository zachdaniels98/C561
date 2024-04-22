import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1  # Including input and output layers

        # Initialize weights and biases
        self.weights = [np.random.rand(hidden_sizes[l], hidden_sizes[l-1]) for l in range(len(hidden_sizes))]
        self.biases = [np.zeros((hidden_sizes[l], 1)) for l in range(len(hidden_sizes))]
        self.weights.append(np.random.rand(output_size, hidden_sizes[-1]))
        self.biases.append(np.zeros((output_size, 1)))

    def forward_propagation(self, X):
        activations = [X]
        for l in range(self.num_layers):
            Z = np.dot(self.weights[l], activations[-1]) + self.biases[l]
            A = self.sigmoid(Z)  # Use sigmoid activation function
            activations.append(A)
        return activations

    def backward_propagation(self, X, y, activations, learning_rate):
        m = X.shape[1]  # Number of training examples

        # Compute output layer error
        error_output = activations[-1] - y
        delta_output = error_output * activations[-1] * (1 - activations[-1])

        # Backpropagate the error
        for l in range(self.num_layers - 2, -1, -1):
            error_hidden = np.dot(self.weights[l+1].T, delta_output)
            delta_hidden = error_hidden * activations[l+1] * (1 - activations[l+1])

            # Update weights and biases
            self.weights[l] -= learning_rate * np.outer(delta_hidden, activations[l])
            self.biases[l] -= learning_rate * np.sum(delta_hidden, axis=1, keepdims=True) / m

            delta_output = delta_hidden

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations, learning_rate)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[-1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage:
input_size = 2
hidden_sizes = [3, 4]  # Number of nodes in each hidden layer
output_size = 1
learning_rate = 0.1
epochs = 1000

X_train = np.random.rand(input_size, 10)  # Training data
y_train = np.random.rand(output_size, 10)  # Target outputs

nn = NeuralNetwork(input_size, hidden_sizes, output_size)
nn.train(X_train, y_train, epochs, learning_rate)

# Make predictions
X_test = np.random.rand(input_size, 5)  # Test data
predictions = nn.predict(X_test)
print("Predictions:", predictions)
