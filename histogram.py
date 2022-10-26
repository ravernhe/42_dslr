import argparse
import pandas as pd
import matplotlib.pyplot as plt

def create_histogram(df, course, legend, xlabel, ylabel, ax=None):
        
    if not ax:
        fig, ax = plt.subplots()

    x = df.loc["Gryffindor"][course]
    y = df.loc["Hufflepuff"][course]
    z = df.loc["Ravenclaw"][course]
    a = df.loc["Slytherin"][course]

    ax.hist(x, histtype="stepfilled", align="mid", color="red", alpha=0.5)
    ax.hist(y, histtype="stepfilled", align="mid", color="yellow", alpha=0.5)
    ax.hist(z, histtype="stepfilled", align="mid", color="blue", alpha=0.5)
    ax.hist(a, histtype="stepfilled", align="mid", color="green", alpha=0.5)

    return ax


def histogram(course):
    legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    xlabel = "Grades"
    ylabel = "Numbers of students"


    try:
        df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House").select_dtypes("number").sort_index()
        if "Index" in df:
            df = df.drop('Index', axis=1)
    except:
        raise Exception("File not found or error while opening the file")

    if course != "all" and not course in df:
        raise Exception("Course not found in dataset")

    if course == "all":
        for name in df:
            ax = create_histogram(df, name, legend, xlabel, ylabel)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=name)
            ax.legend(legend, loc='upper right', frameon=False)
    else:
        ax = create_histogram(df, course, legend, xlabel, ylabel)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=course)
        ax.legend(legend, loc='upper right', frameon=False)
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--course", type=str, help="Select course, default all", default="all")
    args = parser.parse_args()
    histogram(args.course)
