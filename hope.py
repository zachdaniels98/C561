import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, hidden_nodes, output_classes):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.output_classes = output_classes
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases for hidden layers
        for layer in range(self.hidden_layers):
            if layer == 0:
                self.weights.append(np.random.randn(self.input_size, self.hidden_nodes))
            else:
                self.weights.append(np.random.randn(self.hidden_nodes, self.hidden_nodes))
            self.biases.append(np.zeros((1, self.hidden_nodes)))
            self.activations.append(np.zeros((1, self.hidden_nodes)))

        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(self.hidden_nodes, self.output_classes))
        self.biases.append(np.zeros((1, self.output_classes)))

    def forward(self, X):
        # Forward pass through hidden layers
        for layer in range(self.hidden_layers):
            self.activations[layer] = np.maximum(0, np.dot(X, self.weights[layer]) + self.biases[layer])
            X = self.activations[layer]

        # Output layer
        output = np.dot(X, self.weights[-1]) + self.biases[-1]
        return output

    def train(self, X, y, learning_rate=0.001, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss (cross-entropy for multi-class classification)
            loss = -np.sum(y * np.log(output + 1e-10)) / len(X)

            # Backpropagation
            delta = output - y
            for layer in range(self.hidden_layers, -1, -1):
                dW = np.dot(self.activations[layer - 1].T, delta) / len(X)
                db = np.sum(delta, axis=0) / len(X)
                delta = np.dot(delta, self.weights[layer - 1].T) * (self.activations[layer - 1] > 0)
                self.weights[layer - 1] -= learning_rate * dW
                self.biases[layer - 1] -= learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Example usage
if __name__ == "__main__":
    # Generate random data for demonstration
    num_samples = 100
    input_size = 3
    output_classes = 3
    X = np.random.randn(num_samples, input_size)
    y = np.random.randint(output_classes, size=num_samples)
    y_one_hot = np.eye(output_classes)[y]

    mlp = MLP(input_size, hidden_layers=3, hidden_nodes=50, output_classes=output_classes)
    mlp.train(X, y_one_hot, learning_rate=0.01, epochs=1000)

    # Make predictions
    test_X = np.random.randn(10, input_size)
    predictions = mlp.predict(test_X)
    print("Predictions:", predictions)
