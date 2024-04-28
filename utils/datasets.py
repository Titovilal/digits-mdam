import scipy.io.matlab as matlab
from matplotlib import pyplot as plt
import numpy as np


def load_dataset():
    """Loads dataset"""
    mat_file = "BigDigits.mat"
    mat = matlab.loadmat(mat_file, squeeze_me=True)
    data = mat["data"]
    labels = mat["labs"] - 1
    new_labels = [0 if label == 0 else 1 for label in labels]
    return data, new_labels


def show_digit(index, data, labels):
    """Shows a digit given the index in the dataset and its label"""
    digit = data[index]
    label = labels[index]
    # Reshape the digit data to its original 28x28 shape
    digit_image = np.reshape(digit, (28, 28)).T
    # Display the digit
    plt.figure(figsize=(3, 3))
    plt.imshow(digit_image, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Label: {label} ({"only zeros" if label == 0 else "other numbers"})')
    plt.show()


def get_incorrect_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    hits = predictions == y_test
    return np.where(hits == False)[0]
