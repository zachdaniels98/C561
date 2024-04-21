import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, hidden_nodes, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.output_size = output_size

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.weights.append(np.random.rand(self.hidden_nodes[0], self.input_size))
        for i in range(1, hidden_layers):
            self.weights.append(np.random.rand(self.hidden_nodes[i], self.hidden_nodes[i - 1]))
            self.biases.append(np.random.rand(self.hidden_nodes[i], 1))

    def forward_propagation(self, x):
        activations = [x]
        for i in range(self.hidden_layers):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            a = self.relu(z)
            activations.append(a)
        output = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        return output, activations

    def backpropagation(self, x, y, output, activations, learning_rate):
        delta = output - y
        for i in range(self.hidden_layers, -1, -1):
            dz = delta if i == self.hidden_layers else np.dot(self.weights[i + 1].T, delta) * self.relu_derivative(activations[i + 1])
            dw = np.dot(dz, activations[i].T)
            db = dz
            print(len(self.weights), i)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            delta = dz

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            for i in range(X.shape[1]):
                x = X[:, i].reshape(-1, 1)
                y_true = y[:, i].reshape(-1, 1)
                output, activations = self.forward_propagation(x)
                self.backpropagation(x, y_true, output, activations, learning_rate)

    def predict(self, X):
        predictions = []
        for i in range(X.shape[1]):
            x = X[:, i].reshape(-1, 1)
            output, _ = self.forward_propagation(x)
            predictions.append(output)
        return np.hstack(predictions)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, a):
        return np.where(a > 0, 1, 0)

# Example usage:
input_size = 4  # Number of input features
hidden_layers = 2  # Number of hidden layers
hidden_nodes = [8, 8]  # Number of nodes in each hidden layer
output_size = 1  # Number of output units
learning_rate = 0.01
epochs = 1000

mlp = MLP(input_size, hidden_layers, hidden_nodes, output_size)

# Generate random input data and labels
X = np.random.rand(input_size, 100)
y = np.random.rand(output_size, 100)

# Train the MLP
mlp.train(X, y, epochs, learning_rate)

# Make predictions
test_X = np.random.rand(input_size, 10)
predictions = mlp.predict(test_X)
print("MLP predictions:", predictions)