import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the MLP.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): Sizes of hidden layers (e.g., [128, 64]).
            output_size (int): Number of output classes (1 in your case).
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.network_depth = len(hidden_sizes) + 2  # Input + hidden + output layers

        # Initialize weights and biases
        self.weights = [self._kaiming_init(l) for l in range(self.network_depth - 1)]
        self.biases = [np.zeros((size, 1)) for size in hidden_sizes + [output_size]]

    def _kaiming_init(self, layer):
        """
        Initialize weights using Kaiming He's method.

        Args:
            layer (int): Layer index.

        Returns:
            np.ndarray: Initialized weights for the layer.
        """
        fan_in = self.hidden_sizes[layer - 1] if layer > 0 else self.input_size
        return np.random.randn(self.hidden_sizes[layer], fan_in) * np.sqrt(2 / fan_in)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (np.ndarray): Input data (shape: (batch_size, input_size)).

        Returns:
            np.ndarray: Predicted output (shape: (batch_size, output_size)).
        """
        # Apply ReLU activation to hidden layers
        hidden_outputs = [x]
        for l in range(self.network_depth - 2):
            z = np.dot(self.weights[l], hidden_outputs[-1]) + self.biases[l]
            hidden_outputs.append(np.maximum(0, z))

        # Apply sigmoid activation to output layer
        output = 1 / (1 + np.exp(-(np.dot(self.weights[-1], hidden_outputs[-1]) + self.biases[-1])))

        return output

    def calculate_loss(self, y_true, y_pred):
        """
        Calculate L2 loss.

        Args:
            y_true (np.ndarray): True labels (shape: (batch_size, output_size)).
            y_pred (np.ndarray): Predicted labels (shape: (batch_size, output_size)).

        Returns:
            float: L2 loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def activate(self, x):
        """
        Apply ReLU activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Activated output.
        """
        return np.maximum(0, x)

# Example usage:
if __name__ == "__main__":
    # Create an MLP with 3 hidden layers (128 neurons each)
    network_config = (4, [128, 128, 128], 1)
    mlp = MLP(*network_config)

    # Generate random input data (batch_size = 10)
    batch_size = 10
    input_data = np.random.rand(batch_size, 4)

    # Forward pass
    output = mlp.forward(input_data)
    print("Predicted output:\n", output)
