import wn.pretraining as ptr
from wn.net import to_

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim import AdamW

from info_nce import InfoNCE

import pickle
import os
from time import perf_counter
from itertools import chain

# You may not have mflow set up, in which case you probably need to remove this.
import mlflow

experiment_name = "washed_net_pretraining"

# Set all the training parameters up here.

# I don't want to type training_parameters a bunch of times...
tp = {
    # Network architecture
    "dim_model": 36,  # Size of many layers in net
    "dim_proj_head": 16,  # Projection head output_size
    # Training parameters
    "batch_size": 1024,  # Batch size
    "learning_rate": 0.0001,  # Learning rate, currently just fixed
    "n_epochs": 15,  # Training epochs,
    "cut_mix_p": 0.2,  # CutMix probability
    # Other stuff
    "note": "Does this even work",
}


# Data setup ------------------

with open("./data/history_tensor_list.pkl", "rb") as f:
    history_data, p1_id, p2_id = pickle.load(f)

with open("./data/history_interface.pkl", "rb") as f:
    history_interface = pickle.load(f)

# Create a dataset
ds = ptr.PretrainingDataset(history_data)

# This is really just to get multiple dataloader workers to work on Windows.
# It's incredibly annoying.
if __name__ == "__main__":

    # MLflow setup ------------------

    # Auth handled with env variables.
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: {experiment_name}")

    # Network setup ------------------

    # Create the cutmix augmenter
    cut_mix = ptr.CutMix(tp["cut_mix_p"])

    base_net = ptr.PretrainingNet(history_interface, tp["dim_model"])
    projection_head = ptr.ProjectionHead(tp["dim_model"], tp["dim_proj_head"])

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_net.to(device)
    projection_head.to(device)
    cut_mix.to(device)

    print(f"Using {device}")

    with mlflow.start_run():

        # Write all the parameters to mlflow:
        mlflow.log_params(tp)

        # Create a dataloader, optimizer, and criterion

        dl = DataLoader(ds, batch_size=tp["batch_size"], shuffle=True, num_workers=3)

        print(f"Training: {len(dl)} batches of size {tp['batch_size']}")

        optimizer = AdamW(
            chain(
                filter(lambda p: p.requires_grad, base_net.parameters()),
                filter(lambda p: p.requires_grad, projection_head.parameters()),
            ),
            lr=tp["learning_rate"],
        )
        criterion = InfoNCE(reduction="sum")

        # For tracking
        big_tick = perf_counter()
        n_training_obs = 0

        for epoch in range(tp["n_epochs"]):

            print(f"Starting epoch {epoch+1 :2} ------")

            # Training

            base_net.train()
            projection_head.train()

            tick = perf_counter()
            running_loss = 0.0
            running_n = 0

            for i, x in enumerate(dl):

                optimizer.zero_grad()

                # Get a batch
                x = to_(x, device)
                x_augmented = {k: cut_mix(v) for k, v in x.items()}

                y = projection_head(base_net(x))
                y_augmented = projection_head(base_net(x_augmented))

                loss = criterion(y, y_augmented)

                loss.backward()
                optimizer.step()

                n_training_obs += y.shape[0]
                running_loss += loss.item()
                running_n += y.shape[0]

                # mlflow logging
                if i % 10 == 9:
                    mlflow.log_metrics(
                        {
                            "train_loss": running_loss / running_n,
                        },
                        step=n_training_obs,
                    )

                # Print
                if i % 100 == 99:
                    print(
                        f"Epoch {epoch + 1}, Batch {i+1 :4}: {running_loss / running_n :.3f} | ",
                        f"{running_n / (perf_counter() - tick) :6.0f} obs/sec | ",
                        f"{perf_counter() - big_tick :.2f} s",
                    )
                    running_loss = 0.0
                    running_n = 0
                    tick = perf_counter()

    # Save the pretrained network
    torch.save(base_net.state_dict(), "./model/pretrained_model.pt")
