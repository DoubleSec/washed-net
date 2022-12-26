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
    """This class is intended to specify data types and names
    for some specific set of variables. The complete method
    facilitates collecting information that will be useful for
    encoding variables later."""
    def __init__(self, type_map: dict):

        self.type_map = {k: (v, None) for k, v in type_map.items()}
        if not all(v in ["numeric", "time", "categorical"] for v in type_map.values()):
            raise ValueError(
                "All specified types must be 'numeric', 'time', or 'categorical'"
            )

    def complete(self, data: pd.DataFrame):

        for col, val in self.type_map.items():

            dt = data[col]

            if val[0] == "numeric":
                self.type_map[col] = (
                    "numeric",
                    (dt.min(), dt.max()),
                    torch.quantile(
                        torch.tensor(dt.to_numpy(), dtype=torch.float),
                        torch.linspace(0, 1, 16),
                    ),
                )

            elif val[0] == "time":
                self.type_map[col] = ("time", "days")

            else:
                self.type_map[col] = (
                    "categorical",
                    {k: i for i, k in enumerate(dt.unique())},
                )

    def __len__(self):
        return len(self.type_map)

    def __getitem__(self, idx):
        return self.type_map[idx]

    def numeric(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "numeric"}

    def time(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "time"}

    def categorical(self):
        return {k: v for k, v in self.type_map.items() if v[0] == "categorical"}


if __name__ == "__main__":

    match_list = [f"../tennis_atp/atp_matches_{year}.csv" for year in range(2000, 2018)]

    matches = prepare_matches(match_list)

    print(matches.head())
