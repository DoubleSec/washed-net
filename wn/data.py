import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

import pprint


def tr(dt, col_name, interface):

    print(f"Encoding {col_name}")

    if interface[col_name][0] == "numeric":

        dt = torch.tensor(dt.to_numpy(), dtype=torch.float).unsqueeze(-1)
        q = interface[col_name][2].unsqueeze(0).expand([dt.shape[0], -1])

        # Create the tensor and fill the completely filled cells
        encoded = torch.as_tensor(q < dt, dtype=torch.float)

        # Find the partially filled cells
        # (I'm guessing the cuteness here is numerically dangerous.)
        part_filled = torch.argmax(1 / (q - dt), dim=1)

        # Calculate the fill values
        fill_values = q[range(q.shape[0]), part_filled].view([dt.shape[0], 1]) - dt

        # Fill the partial values
        encoded[range(encoded.shape[0]), part_filled] = fill_values.squeeze()

        return encoded

    elif interface[col_name][0] == "time":
        return torch.tensor(dt.to_numpy(), dtype=torch.float).unsqueeze(-1)

    else:
        return torch.tensor(
            [interface[col_name][1][x] for x in dt.to_numpy()], dtype=torch.int
        ).unsqueeze(-1)


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
                    dt.to_numpy().mean().item(),
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

    def __repr__(self):
        return f"DataInterface:\n {pprint.pformat(self.type_map)}"


class MatchDataset(Dataset):
    def __init__(self, input_data: dict, y: torch.Tensor):

        super().__init__()

        self.data = input_data
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        x = {k: v[idx, :] for k, v in self.data.items()}
        y = self.y[idx, :]

        return x, y


class MatchHistoryDataset(Dataset):
    """Extended dataset class that additionally returns history."""

    def __init__(
        self,
        input_data: dict,
        y: torch.Tensor,
        history_data: dict,
        p1_id: torch.Tensor,
        p2_id: torch.Tensor,
        history_size: int = 30,
    ):

        super().__init__()

        self.match_data = input_data
        self.y = y
        self.history_data = history_data
        self.p1_id = p1_id.squeeze()  # Ugly
        self.p2_id = p2_id.squeeze()
        self.history_size = history_size

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):

        # First find the player who we're predicting for and our date limits
        p1_id, p2_id = self.p1_id[idx], self.p2_id[idx]
        end_date = self.match_data["days_elapsed_date"].squeeze()[idx] - 1
        start_date = end_date - 365

        # Get the tabular features and the target
        match_x = {k: v[idx, :] for k, v in self.match_data.items()}
        y = self.y[idx, :]

        # Now get the match history.

        p1_history_x, p1_mask = self._find_history(p1_id, start_date, end_date)
        p2_history_x, p2_mask = self._find_history(p2_id, start_date, end_date)

        return match_x, p1_history_x, p1_mask, p2_history_x, p2_mask, y

    def _find_history(self, pid, start_date, end_date):
        """This function handles retrieving match history for players."""

        # This is implemented as 4 searchsorted calls for speed, which is why
        # we needed to sort this data when we constructed it.

        pid_left_bound = torch.searchsorted(self.p1_id, pid, right=False)
        pid_right_bound = torch.searchsorted(self.p1_id, pid, right=True)

        # This is just for convenience.
        match_dates = self.history_data["days_elapsed_date"].squeeze()

        player_match_dates = match_dates[pid_left_bound:pid_right_bound]
        window_left_bound = (
            torch.searchsorted(player_match_dates, start_date, right=False)
            + pid_left_bound
        )
        window_right_bound = (
            torch.searchsorted(player_match_dates, end_date, right=True)
            + pid_left_bound
        )
        # Limit the amount of history we can look at
        effective_left_bound = torch.maximum(
            window_left_bound, window_right_bound - self.history_size
        )

        history_x = {
            k: self._pad_history(v[effective_left_bound:window_right_bound, :])
            for k, v in self.history_data.items()
        }

        pad_mask = torch.zeros(self.history_size, dtype=torch.bool)
        pad_mask[: (window_right_bound - effective_left_bound)] = 1

        return history_x, pad_mask

    def _pad_history(self, x: torch.Tensor):

        return F.pad(x, [0, 0, 0, self.history_size - x.shape[0]])


if __name__ == "__main__":

    match_list = [f"../tennis_atp/atp_matches_{year}.csv" for year in range(2000, 2018)]

    matches = prepare_matches(match_list)

    print(matches.head())
