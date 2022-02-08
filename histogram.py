from srcs.histogram_class import *

def main():
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    xlabel = "Grades"
    ylabel = "Numbers of students"

    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    df = df.sort_index()

    histogram = Histogram(df, "Care of Magical Creatures", legend, xlabel, ylabel)
    histogram.show_histogram()

if __name__ == "__main__":
    main()
