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

    directories = ["2022-09-30-09-13-11-MOEA", "2022-10-26-11-46-27-MOEA"]
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
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto fronts of 30 experiments")

    plt.savefig("pareto-fronts.png", dpi=300)
    plt.close(fig)


    # another version: size of point is scaled on the number of times it appears
    # step 1: frequency, how many times does the exact same combination of accuracy and number of profiles appear?
    frequency_dict = {}
    for i in range(0, len(accuracies)) :
        key = (profiles[i], accuracies[i])
        if key not in frequency_dict :
            frequency_dict[key] = 1
        else :
            frequency_dict[key] += 1

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

    ax.scatter(profiles, accuracies, marker='o', color='blue', alpha=0.3, s=sizes, label='Candidate solution')
    ax.invert_yaxis()

    ax.set_xlabel("Number of different profiles")
    ax.set_ylabel("Accuracy")
    ax.set_title("Candidate solutions on the Pareto fronts of 30 experiments")
    #ax.legend(loc='best')
    ax.legend(loc='upper right', scatterpoints=3, labelspacing=1)

    plt.savefig("pareto-fronts-different-size.png", dpi=300)
    plt.close(fig)

    return

if __name__ == "__main__" :
    sys.exit( main() )
