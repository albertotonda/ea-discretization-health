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
    

    return

if __name__ == "__main__" :
    sys.exit( main() )
