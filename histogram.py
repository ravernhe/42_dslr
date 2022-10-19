import argparse
from srcs.histogram_class import Histogram
import matplotlib.pyplot as plt
import pandas as pd


def histogram(course):
    legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    xlabel = "Grades"
    ylabel = "Numbers of students"

    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    if course != "all" and not course in df:
        raise Exception("Course not found in dataset")
    if "Index" in df:
        df = df.drop('Index', axis=1)
    df = df.sort_index()

    histogram = Histogram(df, legend, xlabel, ylabel)
    histogram.draw_histogram(course)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--course", type=str, help="Select course, default all", default="all")
    args = parser.parse_args()
    histogram(args.course)
