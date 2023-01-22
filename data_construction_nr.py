from wn.data import prepare_matches, DataInterface, tr
import pandas as pd

import torch

import pickle

# PLEASE NOTE: This file creates data with NO RANK INFORMATION. This is clearly worse,
# but it's useful for experimenting.

# PLEASE ALSO NOTE: It's set up by default to write to the same file names as
# the other data construction script. Sorry, it's confusing.

# Initial processing ---------------------------------------------------------

print("Loading match list")
match_list = [f"../tennis_atp/atp_matches_{year}.csv" for year in range(1968, 2018)]
matches = prepare_matches(match_list)

print("Loading players")
players = pd.read_csv("../tennis_atp/atp_players.csv")

# Add days elapsed from 1900
print("Transforming match dates")
matches.tourney_date = pd.to_datetime(matches.tourney_date.astype("str"))
matches["days_elapsed_date"] = (
    matches.tourney_date - pd.to_datetime("19000101")
).dt.days

# Removing missing birthday players for now
print("Transforming birth dates and removing missing")
players.dob = pd.to_datetime(players.dob.astype("str"), errors="coerce")
players = players[~players.dob.isna()].reset_index(drop=True)
players["days_elapsed_dob"] = (players.dob - pd.to_datetime("19000101")).dt.days

# Find last match dates
# TKTK: There's a better way to do this
# players["last_match_date"] = [
#     matches[matches.winner_id.eq(r.player_id) | matches.loser_id.eq(r.player_id)].days_elapsed_date.max()
#     for r in players.itertuples()
# ]

# Remove matches with players with unknown birthdays
matches = matches.loc[
    matches.winner_id.isin(players.player_id) & matches.loser_id.isin(players.player_id)
].reset_index(drop=True)


# Various Transformations ----------------------------------------------------

# We're basically getting two views of each match here: one from the winner's
# perspective, and one from the loser's.

desired_cols = [
    "best_of",
    "round",
    "winner_hand",
    "loser_hand",
    "surface",
    "tourney_level",
    "days_elapsed_date",
]

augmented_matches = (
    matches.merge(players, "inner", left_on="winner_id", right_on="player_id")
    .loc[:, desired_cols + ["days_elapsed_dob", "loser_id", "winner_id"]]
    .rename({"days_elapsed_dob": "winner_dob"}, axis=1)
    .merge(players, "inner", left_on="loser_id", right_on="player_id")
    .loc[:, desired_cols + ["winner_dob", "days_elapsed_dob", "winner_id", "loser_id"]]
    .rename({"days_elapsed_dob": "loser_dob"}, axis=1)
)

winner_matches = (
    augmented_matches[
        [
            "winner_id",
            "winner_hand",
            "winner_dob",
            "loser_id",
            "loser_hand",
            "loser_dob",
            "surface",
            "tourney_level",
            "best_of",
            "round",
            "days_elapsed_date",
        ]
    ]
    .fillna(-1)
    .assign(won=1)
    .rename(
        {
            "winner_id": "p1_id",
            "winner_hand": "p1_hand",
            "winner_dob": "p1_dob",
            "loser_id": "p2_id",
            "loser_hand": "p2_hand",
            "loser_dob": "p2_dob",
        },
        axis=1,
    )
)

loser_matches = (
    augmented_matches[
        [
            "loser_id",
            "loser_hand",
            "loser_dob",
            "winner_id",
            "winner_hand",
            "winner_dob",
            "surface",
            "tourney_level",
            "best_of",
            "round",
            "days_elapsed_date",
        ]
    ]
    .fillna(-1)
    .assign(won=0)
    .rename(
        {
            "loser_id": "p1_id",
            "loser_hand": "p1_hand",
            "loser_dob": "p1_dob",
            "winner_id": "p2_id",
            "winner_hand": "p2_hand",
            "winner_dob": "p2_dob",
        },
        axis=1,
    )
)

# The sort here is super important. It allows us to do relatively optimized
# data construction on the fly while training the model, later.

condensed_matches = (
    pd.concat([winner_matches, loser_matches])
    .sort_values(["p1_id", "days_elapsed_date"])
    .reset_index(drop=True)
)

# Data Interface Construction ------------------------------------------------

# This is how we control transformations and the network data flow.

# Construct the target match's interface
print("Processing target match features")

match_interface = DataInterface(
    {
        "p1_hand": "categorical",
        "p1_dob": "time",
        "p2_hand": "categorical",
        "p2_dob": "time",
        "surface": "categorical",
        "tourney_level": "categorical",
        "best_of": "categorical",
        "round": "categorical",
        "days_elapsed_date": "time",
    }
)

match_interface.complete(condensed_matches)

# Save the interface
with open("data/match_interface.pkl", "wb") as f:
    pickle.dump(match_interface, f)

# Create the target match data

input_data = {
    k: tr(condensed_matches[k], k, match_interface) for k in match_interface.type_map
}

# Also save the labels
y = torch.tensor(condensed_matches.won.to_numpy(), dtype=torch.float).unsqueeze(1)

# Save the tensor dict and labels
with open("data/tensor_list.pkl", "wb") as f:
    pickle.dump((input_data, y), f)


# Construct the match history interface
print("Processing historical match features")

history_interface = DataInterface(
    {
        "p2_hand": "categorical",
        "p2_dob": "time",
        "surface": "categorical",
        "tourney_level": "categorical",
        "best_of": "categorical",
        "round": "categorical",
        "days_elapsed_date": "time",
        "won": "categorical",
    }
)

history_interface.complete(condensed_matches)

# Encode data to save
input_data = {
    k: tr(condensed_matches[k], k, history_interface)
    for k in history_interface.type_map
}

# And the player IDs
pid_1 = torch.tensor(condensed_matches.p1_id, dtype=torch.int).unsqueeze(1)
pid_2 = torch.tensor(condensed_matches.p2_id, dtype=torch.int).unsqueeze(1)

# Save everything

with open("data/history_tensor_list.pkl", "wb") as f:
    pickle.dump((input_data, pid_1, pid_2), f)

with open("data/history_interface.pkl", "wb") as f:
    pickle.dump(history_interface, f)
