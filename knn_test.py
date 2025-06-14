import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from KNN_implementation import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# print(X_train)
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# print(predictions)

acc= np.sum(predictions==y_test)/len(y_test)
print(f"Accuracy = {acc} %")