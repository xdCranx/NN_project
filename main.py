
# 1. Utworzenie/wprowadzenie do baz przykładów uczących

# 2. Zdefiniowanie rozmiaru sieci

# 3. Ustawienie parametrów uczenia się sieci

# 4. Inicjalizacja wstępnej macierzy wag

import numpy as np
import matplotlib.pyplot as plt


def show_letters():
    # Load data from the CSV file
    csv_file = './MNIST_data/letters_data_set.csv'
    data = np.genfromtxt(csv_file, delimiter=',', dtype=int)

    # Reshape the data into 7x5 arrays
    num_images, image_height, image_width = data.shape[0], 7, 5
    images = data.reshape(num_images, image_height, image_width)

    # Display the images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def load_letters_set():
    data = np.genfromtxt('MNIST_data/letters_data_set.csv', delimiter=',', dtype=int)
    return data


def load_train_set():
    data = np.genfromtxt('MNIST_data/mnist_train.csv', delimiter=',', dtype=int)
    return data


def load_test_set():
    data = np.genfromtxt('MNIST_data/mnist_test.csv', delimiter=',', dtype=int)
    return data


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


load_train_set()