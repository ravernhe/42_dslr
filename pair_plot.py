import matplotlib.pyplot as plt
import pandas as pd
from histogram import create_histogram
from scatter_plot import create_scatter

def pair_plot():
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    try:
        df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House").select_dtypes("number")
        if "Index" in df:
            df = df.drop('Index', axis=1)
    except:
        raise Exception("File not found or error while opening the file")


    size = len(df.columns)  
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(20, 13))

    for y, ycourses in enumerate(df.columns):
        for x, xcourses in enumerate(df.columns):
            if xcourses == ycourses:
                axs[y, x] = create_histogram(df, xcourses, legend, "Grades", "Numbers of students", axs[x, y]) 
            else:
                axs[y, x] = create_scatter(df, legend, xcourses, ycourses, axs[y, x])
            axs[y, x].set_xlabel(xlabel=xcourses, rotation=65)
            axs[y, x].set_ylabel(ylabel=ycourses, rotation=-25, labelpad=60)
            axs[y, x].label_outer()

    fig.legend(legend, loc="upper center")
    plt.show()
        

if __name__ == "__main__" :
    pair_plot()