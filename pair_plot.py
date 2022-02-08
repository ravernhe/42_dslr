from srcs.describe_class import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Check the correlation coeeficient using numpy

def pair_plot_hist(ax, X, indexX, indexY):
    h1 = X[:326].to_numpy()
    ax[indexX, indexY].hist(h1, alpha=0.5)

    h2 = X[326:855].to_numpy()
    ax[indexX, indexY].hist(h2, alpha=0.5)

    h3 = X[855:1298].to_numpy()
    ax[indexX, indexY].hist(h3, alpha=0.5)

    h4 = X[1298:].to_numpy()
    ax[indexX, indexY].hist(h4, alpha=0.5)

def pair_plot_scatter(ax, X, y, indexX, indexY):
    ax[indexX, indexY].scatter(X[:326], y[:326], s=1, color='red', alpha=0.5)
    ax[indexX, indexY].scatter(X[326:855], y[326:855], s=1, color='yellow', alpha=0.5)
    ax[indexX, indexY].scatter(X[855:1298], y[855:1298], s=1, color='blue', alpha=0.5)
    ax[indexX, indexY].scatter(X[1298:], y[1298:], s=1, color='green', alpha=0.5)

def main():
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    rows = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
    # described = Describe("./datasets/dataset_train.csv")
    # described.explain()

    # Try re-open like a donkey
    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    df = df.sort_index()
    # print(df)
    # index out
    # del rows[0]
    size = len(rows)
    fig, ax = plt.subplots(nrows=size, ncols=size, figsize =(18,12))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    x = 0
    for j in rows:
        y = 0
        for i in rows:
            # if feature on same feature histo, else scatter
            if i == j :
                pair_plot_hist(ax, df[i], x, y)
            else :
                pair_plot_scatter(ax, df[i], df[j], x, y)
            
            # Clean graph
            if ax[x, y].get_subplotspec().is_last_row():
                ax[x, y].set_xlabel(i.replace(' ', '\n'))
            else:
                ax[x, y].tick_params(labelbottom=False)

            if ax[x, y].get_subplotspec().is_first_col():
                ax[x, y].set_ylabel(j.replace(' ', '\n'))
            else:
                ax[x, y].tick_params(labelleft=False)

            ax[x, y].spines['right'].set_visible(False)
            ax[x, y].spines['top'].set_visible(False)

            y += 1
        x += 1

    plt.legend(legend, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
    plt.show()
        

if __name__ == "__main__" :
    main()