from math import sqrt
import pandas as pd

def open_set(file_name): # handle error
    return pd.read_csv(file_name)

def main(): #file_name as param : dataset_train.csv
    df = open_set("./datasets/dataset_train.csv")

    # Vérification
    # perc =[.25, .50, .75]
    # include =['float', 'int']
    # desc = df.describe(percentiles = perc, include = include)
    # print(desc, "\n\n")
    # print(df.axes) //Get column name

    print(f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |{"Max":>12}\n')    
    for columns in df.select_dtypes("number").columns.tolist() :
        # Base describe
        print(f'{columns:15.15}', end=' |')
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
        # print("[count]", count, "[Mean]", mean, "[std]", std,"[Min]", min, "[.25]", first_quantile, "[.50]", second_quantile, "[.75]", thrid_quantile, "[Max]", max)
        print(f'{count:>12.5f}', end=' |')
        print(f'{mean:>12.5f}', end=' |')
        print(f'{std:>12.5f}', end=' |')
        print(f'{min:>12.5f}', end=' |')
        print(f'{first_quantile:>12.5f}', end=' |')
        print(f'{second_quantile:>12.5f}', end=' |')
        print(f'{thrid_quantile:>12.5f}', end=' |')
        print(f'{max:>12.5f}', end=' |\n')
    
    # desc.count()
    return 0

if __name__ == "__main__":
    main()