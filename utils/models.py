from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# region Linear
# ------------------------------------------------
from sklearn.linear_model import LinearRegression


def load_linear():
    return LinearRegression()


def fit_linear(linear_model, X_train, y_train):
    return linear_model.fit(X_train, y_train)


def evaluate_linear_f1(linear_model, X_test, y_test, threshold=0.5):
    linear_pred = linear_model.predict(X_test)
    linear_pred = np.where(np.array(linear_pred) <= threshold, 0, 1)
    linear_score = f1_score(y_test, linear_pred)
    return linear_score


# region Quadratic
# ------------------------------------------------
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def load_qda():
    return QuadraticDiscriminantAnalysis()


def fit_qda(qda_model, X_train, y_train):
    return qda_model.fit(X_train, y_train)


# TODO Revisar
def evaluate_qda_f1(qda_model, X_test, y_test):
    qda_pred = qda_model.predict(X_test)
    qda_pred = np.where(np.array(qda_pred) <= 0.5, 0, 1)
    qda_score = f1_score(y_test, qda_pred)
    return np.mean(qda_score)


# region KNN
# ------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier


def load_knn(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def fit_knn(knn_model, X_train, y_train):
    return knn_model.fit(X_train, y_train)


def evaluate_knn_f1(knn_model, X_test, y_test):
    knn_pred = knn_model.predict(X_test)
    knn_score = f1_score(y_test, knn_pred)
    return np.mean(knn_score)


# region SVM
# ------------------------------------------------
from sklearn.svm import SVC


def load_svm():
    return SVC(kernel="linear", random_state=42)


def fit_svm(svm_model, X_train, y_train):
    return svm_model.fit(X_train, y_train)


# TODO Revisar
def evaluate_svm_accuracy(svm_model, X_test, y_test):
    svm_pred = svm_model.predict(X_test)
    svm_score = accuracy_score(y_test, svm_pred)
    return svm_score


# TODO Revisar
def evaluate_svm_f1(svm_model, X_test, y_test):
    svm_pred = svm_model.predict(X_test)
    svm_score = f1_score(y_test, svm_pred)
    return svm_score


# region MLP
# ------------------------------------------------
from sklearn.neural_network import MLPClassifier


def load_mlp(hidden_layer_sizes=(10,), activation="relu", max_iter=200):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter)


def fit_mlp(mlp_model, X_train, y_train):
    return mlp_model.fit(X_train, y_train)


def evaluate_mlp_f1(mlp_model, X_test, y_test):
    mlp_pred = mlp_model.predict(X_test)
    mlp_score = f1_score(y_test, mlp_pred)
    return mlp_score


# region KMeans
# ------------------------------------------------
from sklearn.cluster import KMeans


def load_kmeans(n_clusters=2):
    return KMeans(n_clusters=n_clusters)


def fit_kmeans(kmeans_model, X_train):
    return kmeans_model.fit(X_train)


def evaluate_kmeans_f1(kmeans_model, X_test, y_test):
    kmeans_pred = kmeans_model.predict(X_test)
    kmeans_score = f1_score(y_test, kmeans_pred)
    return kmeans_score
