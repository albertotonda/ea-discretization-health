# Simple script to test a decision tree on the problem

import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

def main() :

    data_file = "../data/data_0.csv"
    labels_file = "../data/labels.csv"
    feature_names_file = "../data/features_0.csv"
    output_figure = "tree.png"

    max_depth = 3
    normalize_data = True

    # read feature names
    feature_names = []
    with open(feature_names_file, "r") as fp :
        feature_names = [f [:-1] for f in fp.readlines() ]
    print(feature_names)

    # read all data
    X = np.genfromtxt(data_file, delimiter=',')
    y = np.genfromtxt(labels_file, delimiter=',')

    # normalize data?
    if normalize_data :
        X = MinMaxScaler().fit_transform(X)

    # try a decision tree (with low depth?)
    print("Training classifier...")
    classifier = DecisionTreeClassifier(max_depth=max_depth)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print("Final accuracy score with a decision tree of depth %d: %.4f" % (max_depth, accuracy))

    print("Saving tree figure as \"%s\"..." % output_figure)
    fig = plt.figure()
    plot_tree(classifier, feature_names=feature_names, max_depth=max_depth, filled=True)
    plt.savefig(output_figure, dpi=300)
    plt.close(fig)

    print("Trying another way to display tree figure...")
    import graphviz

    dot_data = export_graphviz(classifier, out_file=None,
                                feature_names=feature_names,
                                filled=True)
    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

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
