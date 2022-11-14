# Simple script to extract a few meaningful sets of thresholds
import os
import pandas as pd
import sys

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

    print("Index(es) with highest accuracy:", index_highest_accuracy)
    print("Index(es) with lowest number of profiles:", index_lowest_profiles)

    print("Row with highest accuracy:", df.iloc[index_highest_accuracy])
    print("Row with lowest number of profile:", df.iloc[index_lowest_profiles])

    # save the CSV file
    df_selected = df.iloc[[index_highest_accuracy, index_lowest_profiles]]
    df_selected.to_csv("extremes-pareto-front.csv", index=False)

    return

if __name__ == "__main__" :
    sys.exit( main() )
