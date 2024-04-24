from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
# ------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

def load_knn_model(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def fit_knn_model(knn_model, X_train, y_train):
    return knn_model.fit(X_train, y_train)

def evaluate_knn_model(knn_model, X_test, y_test):
    knn_score = cross_val_score(knn_model, X_test, y_test, cv=5)
    return np.mean(knn_score)

# ------------------------------------------------
from sklearn.linear_model import LinearRegression

def load_linear_model():
    return LinearRegression()

def fit_linear_model(linear_model, X_train, y_train):
    return linear_model.fit(X_train, y_train)

def evaluate_linear_model(linear_model, X_test, y_test):
    linear_pred = linear_model.predict(X_test)
    linear_pred = np.where(np.array(linear_pred) <= 0.5, 0, 1)
    linear_score = accuracy_score(y_test, linear_pred)
    return linear_score

# ------------------------------------------------
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def load_qda_model():
    return QuadraticDiscriminantAnalysis()

def fit_qda_model(qda_model, X_train, y_train):
    return qda_model.fit(X_train, y_train)

def evaluate_qda_model(qda_model, X_test, y_test):
    qda_pred = qda_model.predict(X_test)
    qda_pred = np.where(np.array(qda_pred) <= 0.5, 0, 1)
    qda_score = accuracy_score(y_test, qda_pred)
    return np.mean(qda_score)
