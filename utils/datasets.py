import scipy.io.matlab as matlab


def load_dataset():
    """Loads dataset"""
    mat_file = "BigDigits.mat"
    mat = matlab.loadmat(mat_file, squeeze_me=True)
    data = mat["data"]
    labels = mat["labs"] - 1
    new_labels = [0 if label == 0 else 1 for label in labels]
    return data, new_labels
