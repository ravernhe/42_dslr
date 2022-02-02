from typing import Counter
import pandas as pd

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
    df = open_set("../datasets/dataset_train.csv")

    desc = describe(df)
    # desc.count()
    return 0

if __name__ == "__main__":
    main()