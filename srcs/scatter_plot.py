import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("./datasets/dataset_train.csv", index_col = "Hogwarts House")
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    
    i = 0
    # Order data by house and split
    df = df.sort_index()
    test = 1
    for columns in df.select_dtypes("float").columns.tolist() :
        if i < test:
            pass
        elif i == test:
            cpy_columns = columns
        else :
            # x = df[columns][:326].to_numpy()
            # y = df[columns][326:855].to_numpy()
            # z = df[columns][855:1298].to_numpy()
            # a = df[columns][1298:].to_numpy()
    
            plt.scatter(df[columns][:326], df[cpy_columns][:326], color='red', alpha=0.5)
            plt.scatter(df[columns][326:855], df[cpy_columns][326:855], color='yellow', alpha=0.5)
            plt.scatter(df[columns][855:1298], df[cpy_columns][855:1298], color='blue', alpha=0.5)
            plt.scatter(df[columns][1298:], df[cpy_columns][1298:], color='green', alpha=0.5)

            plt.legend(legend, loc='upper right', frameon=False)
            plt.xlabel(columns)
            plt.ylabel(cpy_columns)
            plt.show()
        i += 1
    
    

if __name__ == "__main__":
    main()
