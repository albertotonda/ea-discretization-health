# New version of the script, where we also try to minimize diversity in patient profiles as a secondary fitness

import cma
import copy
import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

def discretize(discretization, X_original) :

    X_discretized = copy.deepcopy(X_original)

    for i in range(0, X_discretized.shape[0]) :
        for j in range(0, X_discretized.shape[1]) :
            value = X_discretized[i,j]
            if value <= discretization[j] :
                value = 0
            elif value > discretization[j] :
                value = 1
            X_discretized[i,j] = value

    return X_discretized

def fitness_function(discretization, X_original, y, verbose=False) : 

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
    performance = []
    for fold_index, train_index, test_index in indexes :

        #print("Working on fold #%d" % fold_index)

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        for classifier, classifier_name in classifierList :

            classifier.fit(X_train, y_train)
            y_test_pred = classifier.predict(X_test)

            #performance.append(f1_score(y_test, y_test_pred))
            performance.append(accuracy_score(y_test, y_test_pred))

    fitness_accuracy = np.mean(performance)

    if verbose == True :
        print("Mean performance is %.4f" % fitness_accuracy)
        print("Performance:", performance)

    # check how many different rows are created by the discretization process
    # and use it as a second fitness
    unique_rows = np.unique(X, axis=0)
    fitness_diversity = unique_rows.shape[0]
    if verbose == True : 
        print("After discretization, there are %d unique rows in the dataset" % unique_rows.shape[0])
     
    # compute total (weighted) fitness
    fitness = 1.0 / (1.0 + fitness_accuracy) + fitness_diversity * 1e-5 # small weight, to make it (hopefully) much less relevant 

    return fitness

def main() :

    # uncomment to set a seed
    seed = None
    # seed = 701607 # 0.99 accuracy and 52 different rows
    seed = 756446 # 0.99 accuracy and 57 different rows

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
    options = {'bounds': [0, 1], 'popsize': 100} # boundaries between 0 and 1, size of the population set to 100
    if seed is not None :
        options["seed"] = seed
    es = cma.CMAEvolutionStrategy(Dimension * [0.5], 0.1, options)

    while not es.stop():
        candidate_solutions = es.ask()
        es.tell(candidate_solutions, [fitness_function(x, data, labels) for x in candidate_solutions])
        es.logger.add()
        es.disp() 
    es.result_pretty()
    x_best = es.result[0]

    # print out the result
    print("Best result:", x_best)
    fitness = fitness_function(x_best, data, labels, verbose=True)

    # print out the best individual
    print("Best individual:")
    for i in range(0, x_best.shape[0]) :
        print(x_best[i], end=" ")
    print()

    return

if __name__ == "__main__" :
    sys.exit( main() )
