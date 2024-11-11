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

# Funcția de activare ReLU
# permite retelei sa invete si sa modeleze relatii complexe intre date
def relu(x):
    return np.maximum(0, x)

# Derivata funcției ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Funcția softmax pentru ieșirea rețelei pentru a obtine probabilitati pentru fiecare clasa
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilizare pentru valori mari
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Funcția de calcul al erorii (cross-entropy loss)
# Masoara cat de bine se potrivest predictiile modelului cu valorile reale
def compute_loss(y_hat, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_hat + 1e-8)) / m  # Stabilizare numerica
    return loss

# Dropout
def dropout(A, dropout_rate):
    keep_prob = 1 - dropout_rate
    mask = np.random.rand(*A.shape) < keep_prob
    A_dropout = A * mask
    A_dropout /= keep_prob

    return A_dropout

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, dropout_rate=0.1, do_dropout=True):
    # Layer 1: Input -> Hidden
    Z1 = np.dot(X, W1) + b1  # Suma ponderată
    A1 = relu(Z1)  # Aplicăm ReLU

    if do_dropout:
        A1 = dropout(A1, dropout_rate)

    # Layer 2: Hidden -> Output
    Z2 = np.dot(A1, W2) + b2  # Suma ponderată
    A2 = softmax(Z2)  # Aplicăm softmax pentru clasificare

    return A1, A2, Z1, Z2

# Backward propagation (folosind chain rule)
def backward_propagation(X, Y, A1, A2, Z1, Z2, W1, b1, W2, b2, learning_rate):
    m = X.shape[0]

    # Derivata pierderii față de ieșirea finala (A2) - Eroare la stratul de ieșire
    dZ2 = A2 - Y  # Eroarea la nivelul ieșirii (diferenta dintre adevarat și prezis)

    # Derivata pierderii fata de W2 și b2
    dW2 = np.dot(A1.T, dZ2) / m  # Derivata funcției de pierdere față de W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Derivata funcției de pierdere față de b2

    # Derivata pierderii fata de A1 (inputul pentru layer-ul 1)
    dA1 = np.dot(dZ2, W2.T)

    # Derivata ReLU față de Z1 - Aplic Chain Rule
    dZ1 = dA1 * relu_derivative(Z1)  # Aplic derivatele ReLU folosind chain rule

    # Derivata pierderii fata de W1 și b1
    dW1 = np.dot(X.T, dZ1) / m  # Derivata funcției de pierdere fata de W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Derivata funcției de pierdere fata de b1

    # Actualizarea greutăților și bias-urilor
    W1 -= learning_rate * dW1  # Actualizăm W1
    b1 -= learning_rate * db1  # Actualizăm b1
    W2 -= learning_rate * dW2  # Actualizăm W2
    b2 -= learning_rate * db2  # Actualizăm b2

    return W1, b1, W2, b2

# Funcție pentru antrenare
def train(X_train, Y_train, X_test, Y_test, learning_rate, epochs, batch_size):
    np.random.seed(42)  # Fixez seminte pentru reproducibilitate
    input_size = X_train.shape[1]  # 784 (pentru MNIST)
    hidden_size = 100  # Numar de neuroni în stratul ascuns
    output_size = 10  # Numar de clase pentru clasificare (pentru MNIST, 10 cifre)

    # Inițializez greutațile și bias-urile
    W1 = np.random.randn(input_size, hidden_size) * 0.01  # Greutați pentru stratul ascuns
    b1 = np.zeros((1, hidden_size))  # Bias pentru stratul ascuns
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # Greutați pentru stratul de ieșire
    b2 = np.zeros((1, output_size))  # Bias pentru stratul de ieșire

    # Antrenare folosind mini-batch
    for epoch in range(epochs):
        # Shuffling datele pentru fiecare epoca
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        # Antrenare pe batch-uri
        for i in range(0, X_train.shape[0], batch_size):
            # Selectăm un batch
            X_batch = X_train_shuffled[i:i + batch_size]
            Y_batch = Y_train_shuffled[i:i + batch_size]

            # Forward propagation
            A1, A2, Z1, Z2 = forward_propagation(X_batch, W1, b1, W2, b2, dropout_rate=0.1, do_dropout=True)

            # Backward propagation
            W1, b1, W2, b2 = backward_propagation(X_batch, Y_batch, A1, A2, Z1, Z2, W1, b1, W2, b2, learning_rate)

        # Calculez loss-ul la fiecare epoca
        A1, A2, _, _ = forward_propagation(X_train, W1, b1, W2, b2, dropout_rate=0.1, do_dropout=False)
        loss = compute_loss(A2, Y_train)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

def main():
    learning_rate = 0.01
    epochs = 100
    batch_size = 20

    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # print(train_X.shape)
    # print(train_Y.shape)

    # Display an image from dataset
    plt.imshow(train_X[9].reshape(28, 28), cmap='gray')
    plt.show()

    # Normalize the pixel values (0-255 -> 0-1)
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # One-hot encode the labels
    # Transform in target value
    encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(10)])
    train_Y_oh = encoder.fit_transform(train_Y.reshape(-1, 1))
    test_Y_oh = encoder.transform(test_Y.reshape(-1, 1))

    # Initialize weights and biases
    np.random.seed(42)
    input_size = train_X.shape[1]  # 784 (for MNIST)
    hidden_size = 100  # Number of neurons in the hidden layer
    output_size = 10  # Number of classes (for MNIST, 10 digits)

    W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for hidden layer
    b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
    W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for output layer
    b2 = np.zeros((1, output_size))  # Bias for output layer

    # Train the network
    W1, b1, W2, b2 = train(train_X, train_Y_oh, test_X, test_Y_oh, learning_rate, epochs, batch_size)

    # Predict and calculate accuracy on the test set
    _, A2, _, _ = forward_propagation(test_X, W1, b1, W2, b2)
    y_test_pred = np.argmax(A2, axis=1)
    test_accuracy = accuracy_score(test_Y, y_test_pred)
    print(f'Final Test Accuracy after {epochs} epochs: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()
