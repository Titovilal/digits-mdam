from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve

class AbstractModel:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_f1(self, X_test, y_test):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def roc_curve(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        return fpr, tpr, thresholds



# region Linear
class Linear(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def evaluate_f1(self, X_test, y_test, threshold=0.5):
        linear_pred = self.model.predict(X_test)
        linear_pred = np.where(np.array(linear_pred) <= threshold, 0, 1)
        linear_score = f1_score(y_test, linear_pred)
        return linear_score


# region QDA
class QDA(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = QuadraticDiscriminantAnalysis()

    def evaluate_f1(self, X_test, y_test):
        qda_pred = self.model.predict(X_test)
        qda_pred = np.where(np.array(qda_pred) <= 0.5, 0, 1)
        qda_score = f1_score(y_test, qda_pred)
        return np.mean(qda_score)


# region KNN
class KNN(AbstractModel):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def evaluate_f1(self, X_test, y_test):
        knn_pred = self.model.predict(X_test)
        knn_score = f1_score(y_test, knn_pred, average="macro")
        return np.mean(knn_score)


# region SVM
class SVM(AbstractModel):
    def __init__(self, C=1, kernel="linear", random_state=None):
        super().__init__()
        self.model = SVC(kernel=kernel, C=C, random_state=random_state)

    def evaluate_f1(self, X_test, y_test):
        svm_pred = self.model.predict(X_test)
        svm_score = f1_score(y_test, svm_pred)
        return svm_score


# region MLP
class MLP(AbstractModel):
    def __init__(self, hidden_layer_sizes=(10,), activation="relu", max_iter=1500):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
        )

    def evaluate_f1(self, X_test, y_test):
        mlp_pred = self.model.predict(X_test)
        mlp_score = f1_score(y_test, mlp_pred)
        return mlp_score

# region KMeans
class KMeans(AbstractModel):
    def __init__(self, n_clusters=2):
        super().__init__()
        self.model = SklearnKMeans(n_clusters=n_clusters)

    def evaluate_f1(self, X_test, y_test):
        kmeans_pred = self.model.predict(X_test)
        kmeans_score = f1_score(y_test, kmeans_pred, average="macro")
        return np.mean(kmeans_score)
