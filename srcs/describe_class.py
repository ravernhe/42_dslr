from math import sqrt
import pandas as pd

class Describe :
    def __init__(self, file_name) -> None:
        self.count = []
        self.mean = []
        self.std = []
        self.min = []
        self.first_quantile = []
        self.second_quantile = []
        self.thrid_quantile = []
        self.max = []
        self.rows = []
        self.file_name = file_name
        self.df = self.open_set()

    def open_set(self): # handle error
        return (pd.read_csv(self.file_name))

    # Function to compare result
    def witness(self):
        perc =[.25, .50, .75]
        include =['float', 'int']
        desc = self.df.describe(percentiles = perc, include = include)
        print(desc, "\n\n")
        print(self.df.axes)

    # init array to compute
    def append_list(self, columns):
        self.count = self.count + [0]
        self.mean = self.mean + [0]
        self.std = self.std + [0]
        self.min = self.min + [0]
        self.first_quantile = self.first_quantile + [0]
        self.second_quantile = self.second_quantile + [0]
        self.thrid_quantile = self.thrid_quantile + [0]
        self.max = self.max + [0]
        self.rows = self.rows + [columns]

    # compute the data
    def explain(self):
        col = 0
        for columns in self.df.select_dtypes("number").columns.tolist() :
            total_sum = 0
            self.append_list(columns)
            for rows in self.df[columns].sort_values():
                # print(rows)
                if pd.notnull(rows) :
                    if (self.count[col] == 0) :
                        self.min[col] = rows
                        self.max[col] = rows
                    if (rows > self.max[col]) :
                        self.max[col] = rows
                    total_sum += rows
                    self.count[col] += 1
            self.mean[col] = total_sum / self.count[col]
            # Quantile life
            first_quantile = .25 * (self.count[col] + 1)
            second_quantile = .50 * (self.count[col] + 1)
            thrid_quantile = .75 * (self.count[col] + 1)
            i = 0
            for rows in self.df[columns].sort_values():
                if pd.notnull(rows) :
                    if i == int(first_quantile) :
                        self.first_quantile[col] = rows
                    if i == int(second_quantile) :
                        self.second_quantile[col] = rows
                    if i == int(thrid_quantile) :
                        self.thrid_quantile[col] = rows
                    i += 1
            # STD PTSD
            std = 0
            for rows in self.df[columns].sort_values():
                if pd.notnull(rows) :
                    std += (rows - self.mean[col]) * (rows - self.mean[col])

            self.std[col] = sqrt(std / (self.count[col] - 1))
            col += 1
    
    # Make it look like describe
    def show(self) :
        print(f'{"":>7}', end = '')
        for name in self.rows :
            if (len(name) > 10) :
                print(f'{name[:9]:>10}', end = '')
                print('...', end = '')
            else :
                print(f'{name:>13}', end = '')

        print("\n", f'{"count":7}', end = '')
        for value in self.count :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"mean":7}', end = '')
        for value in self.mean :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"std":7}', end = '')
        for value in self.std :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"min":7}', end = '')
        for value in self.min :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"25%":7}', end = '')
        for value in self.first_quantile :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"50%":7}', end = '')
        for value in self.second_quantile :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"75%":7}', end = '')
        for value in self.thrid_quantile :
            print(f'{value:>12.5f}', end = ' ')
        print("\n", f'{"max":7}', end = '')
        for value in self.max :
            print(f'{value:>12.5f}', end = ' ')