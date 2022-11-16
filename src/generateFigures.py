# Simple script to plot figures for the paper
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

sns.set_theme() # set seaborn style

def main() :

    #directories = ["2022-09-30-09-13-11-MOEA", "2022-10-26-11-46-27-MOEA"]
    directories = ["2022-11-15-16-55-29-MOEA"]
    all_df = []

    # go over the directories, collect all the data
    for d in directories :
        print("Now analyzing directory \"%s\"..." % d)

        # get the list of interesting files 
        pareto_fronts = [ os.path.join(d, f) for f in os.listdir(d) if f.endswith("-final-pareto-front.csv") ]

        for pf in pareto_fronts :
            all_df.append( pd.read_csv(pf) )

    # merge all dataframes into one
    df = pd.concat(all_df)
    print(df)

    # get the information for the figure
    accuracies = df["accuracy"].values
    profiles = df["number_profiles"].values
    samples = [ [accuracies[i], profiles[i]] for i in range(0, len(accuracies)) ]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(profiles, accuracies, marker='.', color='blue', alpha=0.3)
    
    ax.set_xlabel("Number of different profiles")
    ax.set_ylabel("F1")
    ax.set_title("Pareto fronts of 30 experiments")

    plt.savefig("figures/pareto-fronts.png", dpi=300)
    plt.close(fig)


    # another version: size of point is scaled on the number of times it appears
    # step 0: here we want to do something else, use the expert discretization and evaluate it
    # load data
    dfData = pd.read_csv("../data/data_0.csv", header=None, sep=',')
    data = dfData.values
    dfLabels = pd.read_csv("../data/labels.csv", header=None)
    labels = dfLabels.values.ravel()

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    indexes = [ [index, training, test] for index, [training, test] in enumerate(skf.split(data, labels)) ]

    from MOEAprofileGenerator import discretize, fitness_function
    discretization = [89.8571428571429, 45.0714285714286, 70.8571428571429, 55.5714285714286, 72.0714285714286, 67.5, 0.642857142857143, 4.42857142857143, 24.2857142857143, 7.5, 12.7142857142857, 7230.92857142857]
    fitness, expert_f1, expert_n_profiles = fitness_function(discretization, data, labels, indexes, verbose=True) 
    print("Expert solution: F1=%.4f, n_profiles=%d" % (expert_f1, expert_n_profiles))

    # step 0.5: the same, but the automated discretization found in the previous paper
    discretization = [0.2869, 0.2817, 0.6731, 0.4580, 0.1622, 0.2013, 0.1101, 0.2379, 0.6528, 0.9902, 0.9191, 0.9880]
    # this time, we need to rescale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    fitness, previous_f1, previous_n_profiles = fitness_function(discretization, X_scaled, labels, indexes, verbose=True)
    print("Previous single-objective best solution: F1=%.4f, n_profiles=%d" % (previous_f1, previous_n_profiles))

    # step 1: frequency, how many times does the exact same combination of accuracy and number of profiles appear?
    frequency_dict = {}
    for i in range(0, len(accuracies)) :
        key = (profiles[i], accuracies[i])
        if key not in frequency_dict :
            frequency_dict[key] = 1
        else :
            frequency_dict[key] += 1

    # now, let's prune all results with F1=0.0 (they are not interesting)
    keys_to_remove = [ key for key, value in frequency_dict.items() if key[1] < 0.1 ]
    for key in keys_to_remove : del frequency_dict[key]

    # find highest frequency
    max_frequency = max(frequency_dict.values())

    # modify each value in the dictionary to get a size between 1 and 20
    for key, value in frequency_dict.items() :
        frequency_dict[key] = int(frequency_dict[key] / float(max_frequency) * 19) + 10

    profiles = [ key[0] for key, value in frequency_dict.items() ]
    accuracies = [ key[1] for key, value in frequency_dict.items() ]
    sizes = [ value for key, value in frequency_dict.items() ] 
    print(sizes)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(profiles, accuracies, marker='o', color='blue', alpha=0.3, s=sizes, label='Candidate solutions')
    ax.scatter(expert_n_profiles, expert_f1, marker='^', color='orange', label='Expert-designed solution')
    ax.scatter(previous_n_profiles, previous_f1, marker='x', color='red', label='Best solution in [29]')
    
    ax.invert_yaxis()
    #ax.set_xscale('log')

    ax.set_xlabel("Number of different profiles")
    ax.set_ylabel("F1 (best values bottom-up)")
    ax.set_title("Candidate solutions on the Pareto fronts of 30 experiments")
    #ax.legend(loc='best')
    ax.legend(loc='upper right', scatterpoints=1)

    plt.savefig("figures/pareto-fronts-different-size.png", dpi=300)
    plt.close(fig)

    return

if __name__ == "__main__" :
    sys.exit( main() )
