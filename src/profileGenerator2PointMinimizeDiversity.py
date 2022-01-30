# New version of the script, where we also try to minimize diversity in patient profiles as a secondary fitness

import cma
import copy
import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def discretize(discretization, X_original) :

    X_discretized = copy.deepcopy(X_original)

    for i in range(0, X_discretized.shape[0]) :
        for j in range(0, X_discretized.shape[1]) :
            value = X_discretized[i,j]
            if value <= discretization[i] :
                value = 0
            elif value > discretization[i] :
                value = 1
            X_discretized[i,j] = value

    return X_discretized

def fitness_function(discretization, X_original, y) : 

    n_splits = 10
    fitness = 0.0

    # let's prepare the fitness function evaluating accuracy (or F1?)
    classifierList = [
                [LogisticRegression(solver='lbfgs',), "LogisticRegression"],
            ]

    # discretize dataset
    X = discretize(discretization, X_original)

    # prepare everything for the 10-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    indexes = [ [index, training, test] for index, [training, test] in enumerate(skf.split(X, y)) ]

    # iterate over the folds
    for fold_index, train_index, test_index in indexes :

        print("Working on fold #%d" % fold_index)

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

    return fitness

def main() :

    print("Reading data...")

    dfData = pd.read_csv("../data/data_0.csv", header=None, sep=',')
    dfLabels = pd.read_csv("../data/labels.csv", header=None)
    data = dfData.values
    labels = dfLabels.values.ravel()
    Dimension = data.shape[1] # one value per column

    # pre-processing: scale the data between 0 and 1
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    print("Data dimensions: %d rows and %d columns" % (data.shape[0], data.shape[1]))
    print("Dimension for CMA-ES is: %d" % Dimension)

    # setting up CMA-ES
    es = cma.CMAEvolutionStrategy(Dimension * [0.5], 0.01, {'bounds': [0, 1]})
    while not es.stop():
        candidate_solutions = es.ask()
        es.tell(candidate_solutions, [fitness_function(x, data, labels) for x in candidate_solutions])
        es.logger.add()
        es.disp() 
    es.result_pretty()
    x = es.result[0]

    return

if __name__ == "__main__" :
    sys.exit( main() )
