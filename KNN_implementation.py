import numpy as np
from collections import Counter

def eucliean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) **2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [eucliean_distance(x, x_t) for x_t in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]