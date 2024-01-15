import numpy as np
import matplotlib.pyplot as plt

# Inicjalizacja wag i biasów
np.random.seed(42)

input_size = 1
hidden_size = 10
output_size = 1
beta = 1.25

# Inicjalizacja wag
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Inicjalizacja biasów
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))


# Funkcja aktywacji - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-beta*x))


# Funkcja kosztu - Mean Squared Error
def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)


# Propagacja do przodu
def forward_propagation(inputs):
    # Warstwa ukryta
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Warstwa wyjściowa
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(output_input)

    return hidden_output, final_output


# Propagacja wsteczna
def backward_propagation(inputs, targets, hidden_output, final_output):
    global weights_hidden_output
    global bias_output
    global weights_input_hidden
    global bias_hidden

    # Obliczenie gradientu dla warstwy wyjściowej
    output_error = targets - final_output
    output_delta = output_error * final_output * (1 - final_output)

    # Obliczenie gradientu dla warstwy ukrytej
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * hidden_output * (1 - hidden_output)

    # Aktualizacja wag i biasów
    weights_hidden_output += np.dot(hidden_output.T, output_delta)
    bias_output += np.sum(output_delta, axis=0, keepdims=True)

    weights_input_hidden += np.dot(inputs.T, hidden_delta)
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)


# Funkcja treningowa
def train_neural_network(inputs, targets, epochs=1000000, alpha=0.1):
    for epoch in range(epochs):
        hidden_output, final_output = forward_propagation(inputs)
        cost = mean_squared_error(final_output, targets)

        backward_propagation(inputs, targets, hidden_output, final_output)

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    print("Training complete!")
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


# Przygotowanie danych treningowych
inputs = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
targets = np.array([[1.0], [1.32], [1.6], [1.41], [1.01], [0.6], [0.42], [0.2], [0.51], [0.8]])
targets = targets/10
# Uruchomienie treningu
trained_weights_input_hidden, trained_bias_hidden, trained_weights_hidden_output, trained_bias_output \
    = train_neural_network(inputs, targets)


# Generowanie nowych danych testowych
new_inputs = np.linspace(0, 10, 100).reshape(-1, 1)

# Przewidywanie na nowych danych
_, predictions = forward_propagation(new_inputs)

plt.scatter(new_inputs, predictions*10, label='Przewidywane wyniki')
plt.scatter(inputs, targets*10, label='x')
plt.xlabel('Nowe dane testowe')
plt.ylabel('Przewidywane wyniki')
plt.title('Przewidywane wyniki na nowych danych testowych')
plt.legend()
plt.show()