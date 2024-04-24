import scipy.io.matlab as matlab

def load_dataset():
    # Load the database
    mat_file =  "BigDigits.mat"
    mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
    data = mat["data"]      # read feature vectors
    labels = mat["labs"] - 1  # read labels 1..10
    new_labels = [0 if label == 0 else 1 for label in labels]
    return data, new_labels
