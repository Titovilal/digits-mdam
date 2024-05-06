import scipy.io.matlab as matlab
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math


def load_dataset():
    """Loads dataset"""
    mat_file = "BigDigits.mat"
    mat = matlab.loadmat(mat_file, squeeze_me=True)
    data = mat["data"]
    labels = mat["labs"] - 1
    new_labels = [0 if label == 0 else 1 for label in labels]
    return data, new_labels


def show_digit(digit, label):
    """Shows a digit given the index in the dataset and its label"""
    # Reshape the digit data to its original 28x28 shape
    digit_image = np.reshape(digit, (28, 28)).T
    # Display the digit
    plt.figure(figsize=(3, 3))
    plt.imshow(digit_image, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Label: {label} ({"only zeros" if label == 0 else "other numbers"})')
    plt.show()


def show_digits(digits, labels):
    """Shows a grid of digits given a list of digits and their labels"""
    # Calculate the grid size: more columns than rows
    grid_size = int(math.ceil(math.sqrt(len(digits))))
    nrows = grid_size
    ncols = grid_size

    # Create subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

    for i, ax in enumerate(axs.flat):
        if i < len(digits):
            # Reshape the digit data to its original 28x28 shape
            digit_image = np.reshape(digits[i], (28, 28)).T
            # Display the digit
            ax.imshow(digit_image, cmap=plt.cm.gray_r)
            ax.set_title(
                f'Label: {labels[i]} ({"only zeros" if labels[i] == 0 else "other numbers"})'
            )
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def process_new_image(image_path, label):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC).T
    image = cv2.normalize(image, None, 0, 254, cv2.NORM_MINMAX)
    # Invertir los colores
    image = 255 - image
    image = np.reshape(image, (784,))
    return image, label


def get_incorrect_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    incorrect = predictions != y_test
    incorrect_indices = np.where(incorrect)[0]
    X_incorrect = X_test[incorrect_indices]
    y_incorrect = np.array(y_test)[incorrect_indices]
    y_predictions_incorrect = predictions[incorrect_indices]
    return X_incorrect, y_incorrect, y_predictions_incorrect



