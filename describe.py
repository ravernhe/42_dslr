import argparse
import numpy as np
import pandas as pd


class Describe :
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self.create_dataframe()
        self.aggregated_df = pd.DataFrame()
    
    def create_dataframe(self):
        try:
            df = pd.read_csv(self.file_name, index_col = "Hogwarts House").select_dtypes("number").dropna(axis=1, how='all')
            if "Index" in df:
                df = df.drop('Index', axis=1)
            return df
        except:
            raise Exception("File not found or error while opening the file")

    def percentil(self, data, p, len_data):
        data = data.dropna().sort_values(ignore_index=True)
        k = (len_data - 1) * (p / 100)
        f, c = np.floor(k), np.ceil(k)

        if f == c:
            return data[int(k)]

        return data[int(f)] * (c - k) + data[int(c)] * (k - f)

    def agregate(self):
        for col in self.df:
            calc = {"count" : 0, "mean": 0, "std": 0, "min": None, "25%": 0, "50%": 0, "75%": 0, "max": 0}
            for row in self.df[col].dropna().sort_values(ignore_index=True):  
                calc["count"] += 1
                calc["mean"] += row
                calc["max"] = row
                if calc["min"] is None:
                    calc["min"] = row
            
            calc["mean"] /= calc["count"]
            calc["25%"] = self.percentil(self.df[col], 25, calc["count"])
            calc["50%"] = self.percentil(self.df[col], 50, calc["count"])
            calc["75%"] = self.percentil(self.df[col], 75, calc["count"])


            std = 0
            for row in self.df[col].dropna().sort_values(ignore_index=True):  
                std += (row - calc["mean"]) * (row - calc["mean"])
            calc["std"] = (std / (calc["count"] - 1)) ** 0.5
            
            self.aggregated_df[col] = calc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help="Input dataset")
  args = parser.parse_args()
  describe = Describe(args.dataset)
  describe.agregate()
  print(describe.aggregated_df)