import numpy as np
import csv
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        Initialize the MLP with given architecture.
        Args:
            input_size (int): Number of input nodes.
            hidden_layer_sizes (tuple): Number of nodes in each hidden layer.
            output_size (int): Number of output nodes (usually 1 for regression or binary classification).
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, X):
        """
        Forward pass through the network.
        Args:
            X (numpy.ndarray): Input data (shape: [batch_size, input_size]).
        Returns:
            numpy.ndarray: Output predictions (shape: [batch_size, output_size]).
        """
        self.hidden_outputs = []
        A = X
        for i in range(len(self.hidden_layer_sizes)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.hidden_outputs.append(A)
        self.output = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.output = np.round(self.output).astype(int)
        return self.output

    def backward(self, X, y, learning_rate=0.01):
        """
        Backward pass to update weights and biases using backpropagation.
        Args:
            X (numpy.ndarray): Input data (shape: [batch_size, input_size]).
            y (numpy.ndarray): Ground truth labels (shape: [batch_size, output_size]).
            learning_rate (float): Learning rate for weight updates.
        """
        batch_size = X.shape[0]
        error = self.output - y
        print("error: ", error)

        # Backpropagate through output layer
        d_weights_output = np.dot(self.hidden_outputs[-1].T, error) / batch_size
        d_biases_output = np.sum(error, axis=0) / batch_size
        print(d_weights_output)

        # Update weights and biases for output layer
        self.weights[-1] -= learning_rate * d_weights_output
        self.biases[-1] -= learning_rate * d_biases_output

        # Backpropagate through hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1, 0, -1):
            error = np.dot(error, self.weights[i + 1].T) * (self.hidden_outputs[i] > 0)
            d_weights_hidden = np.dot(self.hidden_outputs[i - 1].T, error) / batch_size
            d_biases_hidden = np.sum(error, axis=0) / batch_size
            self.weights[i] -= learning_rate * d_weights_hidden
            self.biases[i] -= learning_rate * d_biases_hidden

    def compute_accuracy(self, X, y):
        """
        Compute accuracy for a given dataset.
        Args:
            X (numpy.ndarray): Input data (shape: [batch_size, input_size]).
            y (numpy.ndarray): Ground truth labels (shape: [batch_size, output_size]).
        Returns:
            float: Accuracy (between 0 and 1).
        """
        predictions = self.forward(X)
        correct_predictions = np.sum(np.round(predictions) == y)
        total_samples = y.shape[0]
        accuracy = correct_predictions / total_samples
        return accuracy
    
    def clip_gradients(self, max_norm=1.0):
        """
        Clip gradients to prevent exploding gradients.
        Args:
            max_norm (float): Maximum allowed gradient norm.
        """
        for i in range(len(self.weights)):
            weight_grad_norm = np.linalg.norm(self.weights_grad[i])
            if weight_grad_norm > max_norm:
                scale_factor = max_norm / (weight_grad_norm + 1e-8)  # Avoid division by zero
                self.weights_grad[i] *= scale_factor

    def train(self, X, y, X_test, y_test, epochs=1000, learning_rate=0.01, max_norm=1.0):
        """
        Train the MLP using backpropagation.
        Args:
            X (numpy.ndarray): Input data (shape: [batch_size, input_size]).
            y (numpy.ndarray): Ground truth labels (shape: [batch_size, output_size]).
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
        """
        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            self.forward(X)
            print(self.forward(X))
            self.backward(X, y, learning_rate)

            self.clip_gradients(max_norm)

            train_accuracy = self.compute_accuracy(X, y)
            train_accuracies.append(train_accuracy)

            test_accuracy = self.compute_accuracy(X_test, y_test)
            test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.4f} - Test Accuracy: {test_accuracy:.4f}")

        # Plot accuracy curves
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy", color="b")
        plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy", color="r")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()


class DataReader:
    def __init__(self, data=1):
        self.data = data
    
    def get_train_data(self):
        return self.read_data_csv('train_data{}.csv'.format(self.data))
    
    def get_train_label(self):
        return self.read_label_csv('train_label{}.csv'.format(self.data))
    
    def get_test_data(self):
        return self.read_data_csv('test_data{}.csv'.format(self.data))
    
    def get_test_label(self):
        return self.read_label_csv('test_label{}.csv'.format(self.data))
    
    def read_data_csv(self, filename):
        csv_file = filename

        # Initialize lists to store relevant columns
        type_column = []
        price_column = []
        bath_column = []
        propertysqft_column = []

        # Process the CSV file using the csv library
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract relevant columns
                type_column.append(row[1])
                price_column.append(row[2])
                bath_column.append(row[3])
                propertysqft_column.append(row[4])

        # Process the TYPE column
        unique_types = np.unique(type_column)
        type_mapping = {word.split()[0]: idx for idx, word in enumerate(unique_types)}

        # Create a new array with numeric TYPE values
        numeric_type_column = np.array([type_mapping[word.split()[0]] for word in type_column], dtype=int)

        # Create the final NumPy array
        input_array = np.column_stack((numeric_type_column, price_column, bath_column, propertysqft_column))

        # Print the final array (you can save it to a file if needed)
        return input_array.astype(float)
    
    def read_label_csv(self, filename):
        csv_file = filename

        # Initialize an empty list to store the values from the single column
        bed_values = []

        # Process the CSV file using the csv library
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                # Assuming the CSV file has only one column, append the value to the list
                bed_values.append(row[0])

        # Create a 1D NumPy array from the single column values
        beds = np.array(bed_values).reshape(len(bed_values), 1)

        return beds.astype(float)

# Example usage:
if __name__ == "__main__":
    input_size = 4
    hidden_layer_sizes = (8, 8)  # Two hidden layers with 8 nodes each
    output_size = 1
    mlp = MLP(input_size, hidden_layer_sizes, output_size)

    # Generate random input and labels for demonstration
    X_train = np.random.rand(100, input_size)
    y_train = np.random.randint(0, 2, size=(100, output_size))
    X_test = np.random.rand(20, input_size)
    y_test = np.random.randint(0, 2, size=(20, output_size))

    dr = DataReader(2)
    X_train = dr.get_train_data()
    y_train = dr.get_train_label()
    X_test = dr.get_test_data()
    y_test = dr.get_test_label()

    value = 301
    # print(X_train[value])
    X_train_standardized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_train = np.array([X_train_standardized[value]])
    y_train = np.array([y_train[value]])
    # print(X_train)
    print(y_train)
    # print(mlp.forward(first_sample))
    # X_test_standardized = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    # Train the MLP and track accuracy
    mlp.train(X_train, y_train, X_test, y_test, epochs=5, learning_rate=0.1)
    # predictions = mlp.forward(X_test_standardized)

    # Print the predictions (you can modify this part based on your specific use case)
    # print("Predictions:")
    # print(predictions)
