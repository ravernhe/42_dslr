import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    
    i = 0
    # Order data by house and split
    df = df.sort_index()
    for columns in df.select_dtypes("number").columns.tolist() :
        if i == 0 :
            pass
        else :
            x = df[columns][:326].to_numpy()
            y = df[columns][326:855].to_numpy()
            z = df[columns][855:1298].to_numpy()
            a = df[columns][1298:].to_numpy()

            plt.hist(x, histtype="stepfilled", align="mid", color="red", alpha=0.5)
            plt.hist(y, histtype="stepfilled", align="mid", color="yellow", alpha=0.5)
            plt.hist(z, histtype="stepfilled", align="mid", color="blue", alpha=0.5)
            plt.hist(a, histtype="stepfilled", align="mid", color="green", alpha=0.5)
            plt.title(columns)
            plt.legend(legend, loc='upper right', frameon=False)
            plt.xlabel("Grades")
            plt.ylabel("Numbers of students")
            plt.show()
        i += 1
    

if __name__ == "__main__":
    main()
