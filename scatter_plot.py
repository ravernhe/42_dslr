import argparse
import pandas as pd
import matplotlib.pyplot as plt


def create_scatter(df, legend, xlabel, ylabel, ax=None):
    
    if not ax:
        fig, ax = plt.subplots()

    ax.scatter(df[xlabel].loc["Gryffindor"], df[ylabel].loc["Gryffindor"], color='red', alpha=0.5)
    ax.scatter(df[xlabel].loc["Hufflepuff"], df[ylabel].loc["Hufflepuff"], color='yellow', alpha=0.5)
    ax.scatter(df[xlabel].loc["Ravenclaw"], df[ylabel].loc["Ravenclaw"], color='blue', alpha=0.5)
    ax.scatter(df[xlabel].loc["Slytherin"], df[ylabel].loc["Slytherin"], color='green', alpha=0.5)

    return ax


def scatter_plot(xlabel, ylabel):
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    df = pd.read_csv("./datasets/dataset_train.csv", index_col= "Hogwarts House")
    df = df.sort_index()
    if not xlabel in df or not ylabel in df:
        raise Exception("Course not found in dataset")
    ax = create_scatter(df, legend, xlabel, ylabel) 
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(legend, loc='upper right', frameon=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xlabel", type=str, help="Select course one", default="Astronomy")
    parser.add_argument("-y", "--ylabel", type=str, help="Select course two", default="Defense Against the Dark Arts")
    args = parser.parse_args()
    scatter_plot(args.xlabel, args.ylabel)
