# Multi-objective version of the evolutionary profile generator
#

import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from random import Random

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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
                [LogisticRegression(solver='lbfgs'), "LogisticRegression"],
                #[PassiveAggressiveClassifier(), "PassiveAggressive"],
                #[SGDClassifier(), "StochasticGradientDescent"],
                #[SVC(), "SupportVectorMachines"],
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

    if verbose == False :
        return 1.0 / (1.0 + fitness_accuracy), fitness_diversity
    else :
        return fitness, fitness_accuracy, fitness_diversity

@inspyred.ec.evaluators.evaluator
def evaluator(candidate, args) :

    X_original = args["X_original"]
    y = args["y"]

    fitness1, fitness2 = fitness_function(candidate, X_original, y)

    return inspyred.ec.emo.Pareto([fitness1, fitness2])

def generator(random, args) :

    Dimension = args["Dimension"]
    candidate = [ random.uniform(0, 1) for i in range(0, Dimension) ]

    return candidate

def observer(population, num_generations, num_evaluations, args) :

    print("Generation %d (%d evaluations): sample individual %s" % (num_generations, num_evaluations, str(population[0])))

    return


def main() :

    # uncomment to set a seed
    seeds = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    # seed = 701607 # popsize=100, 0.99 accuracy and 52 different rows
    # seed = 756446 # popsize=100, 0.99 accuracy and 57 different rows

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
    print("Dimension for NSGA-II is: %d" % Dimension)

    folder_results = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + sys.argv[0][:-3]
    print("Creating folder to save results \"%s\"..." % folder_results)
    if not os.path.exists(folder_results) : os.makedirs(folder_results)

    # setting up NSGA-II
    for seed in seeds :
        
        # initialize pseudo-random number generator
        prng = Random()
        prng.seed(seed)

        print("\nNow starting experiment with seed %d" % seed)
        nsga2 = inspyred.ec.emo.NSGA2(prng)
        nsga2.observer = observer
        nsga2.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation]
        nsga2.terminator = inspyred.ec.terminators.generation_termination
        final_pop = nsga2.evolve(
                generator = generator,
                evaluator = evaluator,
                pop_size = 100,
                num_selected = 200,
                maximize = False,
                bounder = inspyred.ec.Bounder(lower_bound=0, upper_bound=1),
                max_generations = 10,
                
                # all parts below this will be put into the "args" dictionary
                Dimension = Dimension,
                X_original = data,
                y = labels,

                )

        # now, we save the population as a CSV file, but we also save some information for plots
        accuracies = []
        numbers_of_profiles = []
        with open(os.path.join(folder_results, "experiment-%s.csv" % seed), "w") as fp :

            # header
            header = "accuracy,number_profiles,fitness_accuracy,fitness_profiles"
            for i in range(0, Dimension) : header += ",threshold" + str(i)
            header += "\n"
            fp.write(header)

            for individual in final_pop :
                accuracy = (1.0 / individual.fitness[0]) - 1.0

                fp.write(str(accuracy) + "," + str(individual.fitness[1]) + "," + str(individual.fitness[0]) + "," + str(individual.fitness[1]))
                for i in range(0, len(individual.candidate)) : fp.write("," + str(individual.candidate[i]))
                fp.write("\n")

                accuracies.append(accuracy)
                numbers_of_profiles.append(individual.fitness[1])

        # plot a figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(numbers_of_profiles, accuracies, label="Candidate solutions") 
        ax.set_xlabel("Number of different profiles after discretization")
        ax.set_ylabel("Classification accuracy in a 10-fold cross-validation")
        ax.set_title("Experiment with random seed %d" % seed)
        ax.legend(loc='best')
        plt.savefig(os.path.join(folder_results, "experiment-%d.png" % seed), dpi=300)
        plt.close(fig)

        sys.exit(0) # TODO REMOVE THIS

    return

if __name__ == "__main__" :
    sys.exit( main() )
