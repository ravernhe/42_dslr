from srcs.scatter_plot_class import *

def main():
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    df = df.sort_index()

    scatter = Scatter_plot(df, legend, "Astronomy", "Defense Against the Dark Arts")
    scatter.show()

if __name__ == "__main__":
    main()
