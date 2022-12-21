import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


def prepare_matches(match_files: list):

    df = pd.read_csv(match_files[0])

    if len(match_files) > 1:

        for f in match_files[1:]:

            temp = pd.read_csv(f)
            df = pd.concat([df, temp])

        df = df.sort_values("tourney_date")

    return df.reset_index(drop=True)


class DataInterface:

    def __init__(self, type_map):

        self.type_map = {k: (v, None) for k, v in type_map.items()}
        if not all(v in ["numeric", "time", "categorical"] for v in type_map.values()):
            raise ValueError("All specified types must be 'numeric', 'time', or 'categorical'")

    def complete(self, data):

        for col, val in self.type_map.items():

            dt = data[col]
            
            if val[0] == "numeric":
                self.type_map[col] = ("numeric", (dt.min(), dt.max()))

            elif val[0] == "time":
                self.type_map[col] = ("time", "days")

            else:
                self.type_map[col] = ("categorical", {k: i for i, k in enumerate(dt.unique())})

    def __len__(self):
        return len(self.type_map)

    def type_sizes(self):
        return {
            "numeric": sum(v[0] == "numeric" for v in self.type_map.values()),
            "time": sum(v[0] == "time" for v in self.type_map.values()),
            "categorical": sum(v[0] == "categorical" for v in self.type_map.values()),
        }

    def numeric(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "numeric"}

    def time(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "time"}

    def categorical(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "categorical"}



class NumericRegularizer:
    def __init__(self, range: list):
        self.min = range[0]
        self.max = range[1]

    def __call__(self, x):
        if np.isnan(x):
            return -1
        return (x - self.min) / (self.max - self.min)


class CategoricalIndexer:
    def __init__(self):

        self.category_dictionary = dict()
        self.reverse_dictionary = dict()

    def __call__(self, x):

        if x in self.category_dictionary:
            return self.category_dictionary[x]

        else:
            new_idx = len(self.category_dictionary)

            self.category_dictionary[x] = new_idx
            self.reverse_dictionary[new_idx] = x

            return new_idx

    def inverse(self, idx):

        return self.reverse_dictionary[idx]


if __name__ == "__main__":

    match_list = [f"../tennis_atp/atp_matches_{year}.csv" for year in range(2000, 2018)]

    matches = prepare_matches(match_list)

    print(matches.head())
