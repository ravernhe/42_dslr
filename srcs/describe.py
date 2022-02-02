from cmath import nan
from math import sqrt
from typing import Counter
import pandas as pd
import numpy as np

# Source : towardsdatascience.com/ Standard Deviation
# Do :
    # - Count : nb data
    # - Mean : total / count
    # - Std : Standard Deviation
    # - Min : Smallest
    # - 25% : 1er Quartille | 1st Data > 25 1er %
    # - 50% : 1st Data > 50 1er %
    # - 75% : 1st Data > 75 1er %
    # - Max : Biggest

class describe:
    def __init__(self, df):
        self.df = df
        pass

    def count(self) : #int
        print(self.df.count())
        pass

    def mean(self) : #float
        print(self.df.mean())
        pass

    def std(self) :
        print(self.df.std())
        pass

    def min(self) :
        print(self.df.min())
        pass

    def first_quantile(self) :
        print(self.df.quantile(.25))
        pass

    def second_quantile(self) :
        print(self.df.quantile(.50))
        pass

    def third_quantile(self) :
        print(self.df.quantile(.75))
        pass

    def max(self) :
        print(self.df.max())
        pass

def open_set(file_name): # handle error
    return pd.read_csv(file_name)

def main(): #file_name as param : dataset_train.csv
    df = open_set("./datasets/dataset_train.csv")

    # desc = describe(df)
    perc =[.25, .50, .75]
    include =['float', 'int']
    desc = df.describe(percentiles = perc, include = include)
    print(desc)
    # print(df.axes) //Get column name

    
    for columns in df.select_dtypes("number").columns.tolist() :
        # Base describe
        count = 0
        total_sum = 0
        min = 0
        max = 0
        for rows in df[columns].sort_values():
            # print(rows)
            if pd.notnull(rows) :
                if (count == 0) :
                    min = rows
                    max = rows
                if (rows > max) :
                    max = rows
                total_sum += rows
                count += 1
        mean = total_sum / count
        # Quantile life
        first_quantile = .25 * (count + 1)
        second_quantile = .50 * (count + 1)
        thrid_quantile = .75 * (count + 1)
        i = 0
        for rows in df[columns].sort_values():
            if pd.notnull(rows) :
                if i == int(first_quantile) :
                    first_quantile = rows
                if i == int(second_quantile) :
                    second_quantile = rows
                if i == int(thrid_quantile) :
                    thrid_quantile = rows
                i += 1
        # STD PTSD
        std = 0
        for rows in df[columns].sort_values():
            if pd.notnull(rows) :
                std += (rows - mean) * (rows - mean)

        std = sqrt(std / (count - 1))
        print("[count]", count, "[Mean]", mean, "[std]", std,"[Min]", min, "[.25]", first_quantile, "[.50]", second_quantile, "[.75]", thrid_quantile, "[Max]", max)
    
    # desc.count()
    return 0

if __name__ == "__main__":
    main()