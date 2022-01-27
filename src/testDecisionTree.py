# Simple script to test a decision tree on the problem

import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

def main() :

    data_file = "../data/data_0.csv"
    labels_file = "../data/labels.csv"
    feature_names_file = "../data/features_0.csv"

    max_depth = 3

    # read all data
    X = np.genfromtxt(data_file, delimiter=',')
    y = np.genfromtxt(labels_file, delimiter=',')

    # try a decision tree (with low depth?)
    print("Training classifier...")
    classifier = DecisionTreeClassifier(max_depth=max_depth)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print("Final accuracy score with a decision tree of depth %d: %.4f" % (max_depth, accuracy))

    print("Performing a LOOCV:")
    y_pred = []
    loo = LeaveOneOut()
    for index, [train_index, test_index] in enumerate(loo.split(X)) :

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred.append(classifier.predict(X_test)[0])

    print("Final accuracy in a LOOCV: %.4f" % accuracy_score(y, y_pred))

    return

if __name__ == "__main__" :
    sys.exit( main() )
