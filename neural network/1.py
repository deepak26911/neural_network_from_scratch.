import numpy as np

class NeuralNetwork:
    """
    A simple Neural Network implementation from scratch.

    Attributes:
        input_size (int): The number of neurons in the input layer.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of neurons in the output layer.
        weights1 (np.ndarray): Weights connecting the input layer to the hidden layer.
        weights2 (np.ndarray): Weights connecting the hidden layer to the output layer.
        bias1 (np.ndarray): Biases for the hidden layer.
        bias2 (np.ndarray): Biases for the output layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network's layers, weights, and biases.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values (scaled by 0.01) and biases with zeros.
        # Random initialization is crucial to break symmetry and allow different neurons to learn different features.
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias2 = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        """
        The Sigmoid activation function. It squashes the input values between 0 and 1.
        This is a common choice for activation functions in simple networks.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Calculates the derivative of the sigmoid function.
        This is needed during backpropagation to calculate the gradient of the loss function.

        Args:
            x (np.ndarray): The input array (which is the output of the sigmoid function).

        Returns:
            np.ndarray: The derivative of the sigmoid function.
        """
        return x * (1 - x)

    def feedforward(self, X):
        """
        Performs the feedforward pass of the network.
        It computes the network's output for a given input.

        Args:
            X (np.ndarray): The input data.

        Returns:
            tuple: A tuple containing the output of the hidden layer and the final output layer.
        """
        # Calculate hidden layer's activation
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Calculate output layer's activation
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.final_output = self._sigmoid(self.output_input)

        return self.hidden_output, self.final_output

    def backpropagate(self, X, y, learning_rate):
        """
        Performs the backpropagation algorithm.
        It calculates the error and updates the network's weights and biases.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The true labels.
            learning_rate (float): The step size for gradient descent.
        """
        # 1. Calculate the error (difference between predicted and actual)
        output_error = y - self.final_output
        
        # 2. Calculate the gradient of the output layer
        output_delta = output_error * self._sigmoid_derivative(self.final_output)

        # 3. Calculate the error of the hidden layer
        hidden_error = output_delta.dot(self.weights2.T)
        
        # 4. Calculate the gradient of the hidden layer
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # 5. Update the weights and biases
        # Update weights for the connection between hidden and output layer
        self.weights2 += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        # Update weights for the connection between input and hidden layer
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs, learning_rate):
        """
        Trains the neural network using the training data.

        Args:
            X (np.ndarray): The training input data.
            y (np.ndarray): The training true labels.
            epochs (int): The number of times to iterate over the entire dataset.
            learning_rate (float): The step size for gradient descent.
        """
        for i in range(epochs):
            # Perform a feedforward pass
            self.feedforward(X)
            # Perform backpropagation to update weights
            self.backpropagate(X, y, learning_rate)

            # Print the loss (Mean Squared Error) at every 1000 epochs
            if (i + 1) % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))
                print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Makes predictions on new, unseen data.

        Args:
            X (np.ndarray): The input data for prediction.

        Returns:
            np.ndarray: The predicted output, rounded to 0 or 1.
        """
        _, prediction = self.feedforward(X)
        return np.round(prediction)

# --- Example Usage: Solving the XOR problem ---
if __name__ == "__main__":
    # The XOR problem is a classic problem for neural networks because
    # it is not linearly separable.
    
    # Input data (X) for the XOR function
    X_train = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    # Output data (y) for the XOR function
    y_train = np.array([[0],
                        [1],
                        [1],
                        [0]])

    # Create a neural network:
    # - Input layer has 2 neurons (for the 2 inputs)
    # - Hidden layer has 4 neurons (this can be tuned)
    # - Output layer has 1 neuron (for the single binary output)
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the network
    print("Starting training...")
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)
    print("Training finished.\n")
    
    # Make predictions on the training data to see how well it learned
    print("Predictions on training data:")
    predictions = nn.predict(X_train)

    for i in range(len(X_train)):
        print(f"Input: {X_train[i]} -> Predicted: {predictions[i][0]}, Actual: {y_train[i][0]}")

    # Test with new data points
    print("\nTesting with a new data point [0, 0]:")
    new_prediction = nn.predict(np.array([[0, 0]]))
    print(f"Prediction for [0, 0]: {new_prediction[0][0]}")
