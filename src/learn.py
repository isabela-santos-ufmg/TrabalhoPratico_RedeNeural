import numpy as np
np.random.seed(42)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', as_frame=False, cache=True, version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Normalize the data
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)*0.01
    W2 = np.random.randn(hidden_size, output_size)*0.01
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, output_size))
    return (W1, b1, W2, b2)
# Activation functions
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    exp = np.exp(Z)
    return exp/np.sum(exp, axis=-1, keepdims=True) #axis e keepdims pois Z tem duas linhas

# Forward pass
def forward(X, parameters): 
    W1, b1, W2, b2 = parameters
    #parte linear da camada 1 
    Z1 = np.dot(X,W1) + b1
    #funcao de ativaçao camada 1 
    A1 = relu(Z1) 
    #parte linear da camada 2
    Z2 = np.dot(A1,W2) + b2
    #funcao de ativaçao camada 2 (especialmente para casos multiclasse na ultima camada)
    A2 = softmax(Z2)
    return (Z1, A1, Z2, A2)
# Loss function
def cross_entropy_loss(y_true, y_pred):
    # funcao de loss ideal para casos multiclass
    m = y_true.shape[0]
    fraction = -1/m #qtd de linhas de y_true é m:= qtd de dados que temos 
    summation = np.sum(y_true*np.log(y_pred))
    loss = fraction*summation
    return loss
# Backward pass
def backward(X, y, W2, Z1, A1, A2):
    n_samples = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    dA1 = np.dot(dZ2, W2.T)
    
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
    
    return dW1, db1, dW2, db2

def gradient_descent_step(X, y, parameters, learning_rate=0.01):
    W1, b1, W2, b2 = parameters
    (Z1, A1, Z2, A2) = forward(X, parameters)
    loss = cross_entropy_loss(y, A2) # y é o rotulo real enquanto A2 é a saída da rede neural 
    dW1, db1, dW2, db2 = backward(X, y,  parameters[2], Z1, A1, A2) #derivadas parciais 
    W1 = W1 - learning_rate * dW1
    b1  = b1 - learning_rate * db1
    W2  = W2 - learning_rate * dW2
    b2  = b2 - learning_rate * db2
    return (W1, b1, W2, b2), loss
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


# Validação de saída  ----------------------------------
"""
(W1, b1, W2, b2) = initialize_parameters(784, 4, 10) 
pesos = (W1, b1, W2, b2)
#for item in pesos:
#   print(item)
    
a = relu(np.array([[-1, 0, 1], [-2, 0, 2]]))
b = softmax(np.array([[1, 2, 3], [4, 5, 6]]))
#print(a)
#print(b)

X = np.random.randn(2, 784)
Z1, A1, Z2, A2 = forward(X, pesos)

funcoes = (Z1, A1, Z2, A2)
#for item in funcoes:
#   print(item)

y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
loss = cross_entropy_loss(y_true, y_pred)
#print(loss)

X = np.random.randn(2, 784)
y = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
W1, b1, W2, b2 = initialize_parameters(784, 4, 10)
parameters = (W1, b1, W2, b2)

parameters, loss = gradient_descent_step(X, y, parameters, learning_rate=0.01)
W1, b1, W2, b2 = parameters

print("=" * 45)
print(f"Loss: {loss:.10f}")
print("=" * 45)

for nome, arr in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
    print(f"\n[{nome}] shape: {arr.shape}")
    print(arr)

y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
acc = accuracy(y_true, y_pred)
print(acc)

"""


