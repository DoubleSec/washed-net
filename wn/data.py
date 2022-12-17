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


class MatchDataset(Dataset):

    COLS = [
        "winner_seed",
        "winner_hand",
        "loser_seed",
        "loser_hand",
        "surface",
        "tourney_level",
        "round",
        "best_of",
    ]
    COL_TYPES = ["num", "cat", "num", "cat", "cat", "cat", "cat", "cat"]

    def __init__(self, matches):

        super().__init__()
        self.matches = matches[self.COLS]

        self.type_regularizers = dict()

        for col, col_type in zip(self.COLS, self.COL_TYPES):

            if col_type == "num":

                self.type_regularizers[col] = NumericRegularizer(
                    [matches[col].min(), matches[col].max()]
                )

            else:

                self.type_regularizers[col] = CategoricalIndexer()

    def __len__(self):

        return len(self.matches)

    def __getitem__(self, idx):

        row = self.matches.iloc[idx]

        x = torch.zeros([1, len(self.type_regularizers)])

        for i, pair in enumerate(self.type_regularizers.items()):
            x[i, 0] = pair[1](row[pair[0]])

        return x


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
