# Another script

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys

# stuff used for custom color maps
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# adjust layout
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def set_figure_aesthethics(fig, ax, im, index_class_1) :

    ax.set_aspect(0.07)

    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(["ENSG00000198826", "ENSG00000170298", "ENSG00000214548", "ENSG00000287576", "ENSG00000240403", "ENSG00000214174", "ENSG00000214460", "ENSG00000263551", "ENSG00000220785", "ENSG00000224227", "ENSG00000186523", "ENSG00000155657"], fontsize=6)
    ax.tick_params(axis='x', rotation=90)

    #ax.set_yticks([0, index_class_1]) 
    #ax.set_yticklabels(["Mild symptoms", "Severe symptoms"])
    ax.set_yticks([0, index_class_1])
    yticklabels = ["Mild symptoms", "Severe symptoms"]
    ax.set_yticklabels(yticklabels, fontsize=6)

    # let's experiment with 'annotate'
    # this is for severe symptoms
    ax.annotate('', xy=(-0.1, 0), xycoords='axes fraction', xytext=(-0.1, 0.19), arrowprops=dict(arrowstyle="<->", color='r'))
    # this is for mild symptoms
    ax.annotate('', xy=(-0.1, 0.22), xycoords='axes fraction', xytext=(-0.1, 0.98), arrowprops=dict(arrowstyle="<->", color='y'))

    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    fig.colorbar(im, shrink=0.7)

    return ax

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
    df.reset_index(inplace=True)
    print(df)

    # find and select rows: row with the highest accuracy, row with lowest number of profiles
    index_highest_accuracy = df["accuracy"].idxmax()
    index_lowest_profiles = df["number_profiles"].idxmin()
    print("Row with highest accuracy:", df.iloc[index_highest_accuracy])
    print("Row with lowest number of profiles:", df.iloc[index_lowest_profiles])

    # get the candidate solutions
    columns_thresholds = [c for c in df.columns if c.startswith("threshold")]
    solution_highest_accuracy = df.iloc[index_highest_accuracy][columns_thresholds].values
    solution_lowest_profiles = df.iloc[index_lowest_profiles][columns_thresholds].values

    print("Solution with highest accuracy:", solution_highest_accuracy)
    print("Solution with lowest number of profiles:", solution_lowest_profiles)

    # load original data
    data_features = "../data/data_0.csv"
    data_labels = "../data/labels_0.csv"
    df = pd.read_csv("../data/data_0.csv", header=None, sep=',')
    df.columns = [ "feature_%d" % i for i in range(0, len(df.columns)) ]
    df_labels = pd.read_csv("../data/labels.csv", header=None)
    df_labels.columns = ["class"]
    df["class"] = df_labels["class"].values

    # sort dataframe by class label
    df.sort_values(by=["class"], inplace=True)
    df.reset_index(inplace=True)
    print(df)

    data = df[[c for c in df.columns if c.startswith("feature")]].values
    labels = df["class"].values.ravel()

    # rescale data between 0 and 1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # find first index of class value '1'
    index_class_1 = -1
    for index, row in df.iterrows() :
        if index_class_1 == -1 and row["class"] == 1 :
            index_class_1 = index

    from MOEAprofileGenerator import discretize, fitness_function
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    indexes = [ [index, training, test] for index, [training, test] in enumerate(skf.split(data, labels)) ]
    fitness, fitness_accuracy, fitness_diversity = fitness_function(solution_lowest_profiles, data, labels, indexes, verbose=True) 
    fitness, fitness_accuracy, fitness_diversity = fitness_function(solution_highest_accuracy, data, labels, indexes, verbose=True) 

    X_discrete_highest_accuracy = discretize(solution_highest_accuracy, data)
    X_discrete_lowest_profiles = discretize(solution_lowest_profiles, data)

    # plot original figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap='Greens')
    ax.set_title("Normalized gene expression data")
    set_figure_aesthethics(fig, ax, im, index_class_1)
    plt.savefig("figures/original-gene-expressions.png", dpi=300)
    plt.close(fig)

    # custom color map
    newcmp = ListedColormap(['limegreen', 'green'])

    # plot figures
    fig = plt.figure()
    #fig.tight_layout()
    ax = fig.add_subplot(111)

    #ax.imshow(X_discrete_highest_accuracy, cmap=newcmp)
    im = ax.imshow(X_discrete_lowest_profiles, cmap=newcmp)

    # aesthethics
    ax = set_figure_aesthethics(fig, ax, im, index_class_1)
    ax.set_title("Discretized profiles with highest-accuracy solution")

    #fig.align_labels()
    plt.savefig("figures/heatmap-highest-accuracy.png", dpi=300)
    plt.close(fig)


    return

if __name__ == "__main__" :
    sys.exit( main() )
