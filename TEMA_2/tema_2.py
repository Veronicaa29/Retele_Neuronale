import numpy as np
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
    transform=lambda x: np.array(x).flatten(),
    download=True,
    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels

def softmax(X, W, b):
    """
    X: Input matrix (m examples, 784 features)
    W: Weights matrix (784 inputs, 10 outputs)
    b: Bias vector (10 outputs)
    """
    # Compute the weighted sum of inputs and bias
    z = np.dot(X, W) + b
    # Activation function: Compute the softmax of the weighted sum
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement for large z values
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Softmax to get probabilities

    return y_hat

def compute_loss(y_hat, y_true):
    """
    y_hat: Predicted probabilities (m examples, 10 classes)
    y_true: True labels in one-hot encoded format (m examples, 10 classes)
    """
    m = y_true.shape[0]  # Number of examples
    # Compute the cross-entropy loss
    loss = -np.sum(y_true * np.log(y_hat + 1e-8)) / m  # Adding a small value for numerical stability
    return loss

def backward_propagation(X, y_true, y_hat, W, b, learning_rate):
    """
    X: Input matrix (m examples, 784 features)
    y_true: True labels (one-hot encoded, m examples, 10 classes)
    y_hat: Predicted probabilities (m examples, 10 classes)
    W: Weights matrix (784 x 10)
    b: Bias vector (10)
    learning_rate: Step size for gradient descent
    """
    m = X.shape[0]  # Number of examples

    # Step 1: Calculate the error (gradient of the loss)
    error = y_true - y_hat  # Target - y_hat

    # Step 2: Update weights and biases using the formulas provided
    W += learning_rate * np.dot(X.T, error)  # Update weights: W = W + μ × (Target - y) × X^T
    b += learning_rate * np.sum(error, axis=0)  # Update bias: b = b + μ × (Target - y)

    return W, b  # Return updated weights and biases

# def train(X, y_true, W, b, learning_rate, epochs):
#     for i in range(epochs):
#         # Step 1: Forward propagation
#         y_hat = softmax(X, W, b)
#
#         # Step 2: Compute the loss
#         loss = compute_loss(y_hat, y_true)
#
#         # Step 3: Backward propagation
#         W, b = backward_propagation(X, y_true, y_hat, W, b, learning_rate)
#
#         if i % 100 == 0:
#             print(f"Epoch {i}, Loss: {loss}")
#
#     return W, b

def predict(X, W, b):
    """
    Predict the class for each example in the input X
    """
    y_hat = softmax(X, W, b)
    return np.argmax(y_hat, axis=1)

def train_with_batches(X, y_true, W, b, learning_rate, epochs, batch_size):
    """Train the perceptron using mini-batch gradient descent."""
    m = X.shape[0]  # Number of training examples

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        perm = np.random.permutation(m)
        X_shuffled = X[perm]
        y_shuffled = y_true[perm]

        # Split the data into batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward propagation for the batch
            y_hat = softmax(X_batch, W, b)

            # Compute the loss for the batch
            loss = compute_loss(y_hat, y_batch)

            # Backward propagation for the batch
            W, b = backward_propagation(X_batch, y_batch, y_hat, W, b, learning_rate)

        if epoch % 10 == 0:
            y_train_pred = predict(X, W, b)
            acc = accuracy_score(np.argmax(y_true, axis=1), y_train_pred)
            print(f'Epoch {epoch}, Training Accuracy: {acc:.4f}, Loss: {loss}')

    return W, b

def main():
    # # Example initialization
    # np.random.seed(42)  # For reproducibility
    # X = np.random.rand(100, 784)  # 100 examples with 784 features (inputs)
    # y_true = np.eye(10)[np.random.choice(10, 100)]  # 100 random one-hot encoded labels
    #
    # W = np.random.randn(784, 10) * 0.01  # Small random weights
    # b = np.zeros((1, 10))  # Zero biases
    #
    # # Training the perceptron
    # learning_rate = 0.01
    # epochs = 1000
    # W, b = train(X, y_true, W, b, learning_rate, epochs)

    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # print(train_X.shape)
    # print(train_Y.shape)

    # Display an image from dataset
    plt.imshow(train_X[10].reshape(28, 28), cmap='gray')
    plt.show()

    # Normalize the pixel values (0-255 -> 0-1)
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(10)])
    train_Y_oh = encoder.fit_transform(train_Y.reshape(-1, 1))
    test_Y_oh = encoder.transform(test_Y.reshape(-1, 1))

    # Initialize weights and biases
    np.random.seed(42)
    W = np.random.randn(784, 10) * 0.01  # Small random weights
    b = np.zeros((1, 10))  # Zero bias

    # Initial accuracy before training
    y_test_pred_initial = predict(test_X, W, b)
    initial_accuracy = accuracy_score(test_Y, y_test_pred_initial)
    print(f'Initial Test Accuracy: {initial_accuracy:.4f}')

    # Train the network
    learning_rate = 0.01
    epochs = 100  # You can choose between 50-500
    batch_size = 100
    W, b = train_with_batches(train_X, train_Y_oh, W, b, learning_rate, epochs, batch_size)

    # Accuracy after training
    y_test_pred_final = predict(test_X, W, b)
    final_accuracy = accuracy_score(test_Y, y_test_pred_final)
    print(f'Final Test Accuracy after {epochs} epochs: {final_accuracy:.4f}')

if __name__ == "__main__":
    main()