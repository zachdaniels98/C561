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
            self.biases.append(np.random.rand(layer_sizes[i + 1]))

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.
        Args:
            z (numpy.ndarray): Input value.
        Returns:
            numpy.ndarray: Output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Computes the derivative of the sigmoid function.
        Args:
            z (numpy.ndarray): Input value.
        Returns:
            numpy.ndarray: Derivative of the sigmoid function.
        """
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    
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
            A = self.sigmoid(Z)
            self.hidden_outputs.append(A)
        output_Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.output = self.relu(output_Z)
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

        # Backpropagate through output layer
        d_weights_output = np.dot(self.hidden_outputs[-1].T, error) / batch_size
        d_biases_output = np.sum(error, axis=0) / batch_size

        # Update weights and biases for output layer
        self.weights[-1] -= learning_rate * d_weights_output
        self.biases[-1] -= learning_rate * d_biases_output

        # Backpropagate through hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1, 0, -1):
            error = np.dot(error, self.weights[i + 1].T) * self.sigmoid_derivative(self.hidden_outputs[i])
            d_weights_hidden = np.dot(self.hidden_outputs[i - 1].T, error) / batch_size
            d_biases_hidden = np.sum(error, axis=0) / batch_size
            self.weights[i] -= learning_rate * d_weights_hidden
            self.biases[i] -= learning_rate * d_biases_hidden

        pre_rounded_output = np.dot(self.hidden_outputs[-1], self.weights[-1]) + self.biases[-1]
        output_error = pre_rounded_output - y
        d_weights_output = np.dot(self.hidden_outputs[-1].T, output_error) / batch_size
        d_biases_output = np.sum(output_error, axis=0) / batch_size
        self.weights[-1] -= learning_rate * d_weights_output
        self.biases[-1] -= learning_rate * d_biases_output

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

    def train(self, X, y, X_test, y_test, epochs=1000, learning_rate=0.01, batch_size=32):
        """
        Train the MLP using backpropagation.
        Args:
            X (numpy.ndarray): Input data (shape: [batch_size, input_size]).
            y (numpy.ndarray): Ground truth labels (shape: [batch_size, output_size]).
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
        """
        def learning_rate_schedule(epoch):
            # Example: Reduce learning rate by half every 10 epochs
            if epoch % 10 == 0:
                return learning_rate / 2
            else:
                return learning_rate

        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            current_learning_rate = learning_rate_schedule(epoch)

            for batch_start in range(0, X.shape[0], batch_size):
                batch_X = X[batch_start : batch_start + batch_size]
                batch_y = y[batch_start : batch_start + batch_size]

                self.forward(batch_X)
                self.backward(batch_X, batch_y, current_learning_rate)
            
            # validation_accuracy = self.compute_accuracy(X_test, y_test)
            # print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {validation_accuracy:.4f}")

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
        input_array = np.column_stack((price_column, bath_column, propertysqft_column))

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
    input_size = 3
    hidden_layer_sizes = [50, 100, 50]  # Two hidden layers with 8 nodes each
    output_size = 1
    mlp = MLP(input_size, hidden_layer_sizes, output_size)

    dr = DataReader(5)
    X_train = dr.get_train_data()
    y_train = dr.get_train_label()
    X_test = dr.get_test_data()
    y_test = dr.get_test_label()

    # # print(X_train[value])
    X_train_standardized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_train = np.array(X_train_standardized)
    y_train = np.array(y_train)
    # print(X_train)
    # print(y_train)
    # print(mlp.forward(first_sample))
    X_test_standardized = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
    X_test = np.array(X_test_standardized)

    # Train the MLP and track accuracy
    mlp.train(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.0001, batch_size=32)
    # predictions = mlp.forward(X_test_standardized)

    # Print the predictions (you can modify this part based on your specific use case)
    # print("Predictions:")
    # print(predictions)
