from wn import net
from wn.data import MatchHistoryDataset

import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim import AdamW

import pickle
import os
from time import perf_counter

# You may not have mflow set up, in which case you probably need to remove this.
import mlflow

experiment_name = "washed_net"

# Set all the training parameters up here.

# I don't want to type training_parameters a bunch of times...
tp = {
    # Data parameters
    "history_length": 30,  # How many matches of history are provided
    "validation_set_size": 0.25,  # Ratio for train/test split.
    # Network architecture
    "col_encoding_size": 12,  # Size of learned column/position encodings
    "learnable_padding": False,  # Control whether padding embeding is learned
    "dim_model": 48,  # Size of many layers in net
    "dim_ff": 64,  # Size of linear layers in transformer
    "n_transformer_layers": 4,  # Transformer depth
    "n_transformer_heads": 1,  # Attention heads, must divide dim_model
    "n_output_layers": 1,  # Depth of linear output layers
    # Training parameters
    "batch_size": 1024,  # Batch size
    "learning_rate": 0.0001,  # Learning rate, currently just fixed
    "n_epochs": 10,  # Training epochs,
    "use_pretrained": False,  # Use pre-trained layers?
    # Other stuff
    "note": "I just need a trained network to test something",
}


# Data setup -----------------------------------------------------------------

# Load the tensorized data
with open("data/tensor_list.pkl", "rb") as f:
    input_data, y = pickle.load(f)

with open("data/history_tensor_list.pkl", "rb") as f:
    history_data, p1_id, p2_id = pickle.load(f)

# Load the interfaces
with open("data/match_interface.pkl", "rb") as f:
    match_interface = pickle.load(f)

with open("data/history_interface.pkl", "rb") as f:
    history_interface = pickle.load(f)


# Make a dataset
ds = MatchHistoryDataset(
    input_data, y, history_data, p1_id, p2_id, history_size=tp["history_length"]
)

# Split into training and validation
idx = torch.randperm(len(input_data["p1_dob"]))
split_idx = int(idx.shape[0] * tp["validation_set_size"])
train_ds = Subset(ds, idx[split_idx:])
validation_ds = Subset(ds, idx[:split_idx])

# This is really just to get multiple dataloader workers to work on Windows.
# It's incredibly annoying.
if __name__ == "__main__":

    # MLflow setup

    # Auth handled with env variables.
    # If you don't set a tracking uri it should default to a local directory, which
    # should work fine I think, but you'll need to remove this line.
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)
    print(f"Using MLflow experiment: {experiment_name}")

    # Network setup --------------------------------------------------------------

    # Special tabular input layer
    table_input_layer = net.TabularInputLayer(
        interface=match_interface,
        col_encoding_size=tp["col_encoding_size"],
        embedding_size=tp["dim_model"] - tp["col_encoding_size"],
        append_cls=True,
    )

    # Input layer for sequential features, one for each player
    p1_sequence_input_layer = net.SequentialInputLayer(
        interface=history_interface,
        sequence_encoding_size=[tp["history_length"], tp["col_encoding_size"]],
        embedding_size=tp["dim_model"] - tp["col_encoding_size"],
        learnable_padding=tp["learnable_padding"],
    )

    p2_sequence_input_layer = net.SequentialInputLayer(
        interface=history_interface,
        sequence_encoding_size=[tp["history_length"], tp["col_encoding_size"]],
        embedding_size=tp["dim_model"] - tp["col_encoding_size"],
        learnable_padding=tp["learnable_padding"],
    )

    # Load the embeddings and projection layers from the pretrained_model

    if tp["use_pretrained"]:

        # Load the pre-trained parts of this model:
        pretrained_state_dict = torch.load("./model/pretrained_model.pt")
        p1_sequence_input_layer.load_state_dict(pretrained_state_dict, strict=False)
        p2_sequence_input_layer.load_state_dict(pretrained_state_dict, strict=False)

    output_layers = net.OutputLayers(tp["dim_model"], tp["n_output_layers"])

    # Transformer encoder
    tr = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=tp["dim_model"],
            nhead=tp["n_transformer_heads"],
            dim_feedforward=tp["dim_ff"],
            batch_first=True,
            norm_first=True,
        ),
        num_layers=tp["n_transformer_layers"],
    )

    whole_net = net.FusionNet(
        table_input_layer=table_input_layer,
        p1_sequence_input_layer=p1_sequence_input_layer,
        p2_sequence_input_layer=p2_sequence_input_layer,
        transformer=tr,
        output_layer=output_layers,
    )

    n_weights = sum([p.numel() for p in whole_net.parameters() if p.requires_grad])
    print(f"Network has {n_weights} weights.")

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whole_net.to(device)

    print(f"Using {device}")

    # Training -------------------------------------------------------------------

    with mlflow.start_run():

        # Write all the parameters to mlflow:
        mlflow.log_params(tp)
        mlflow.log_param("n_weights", n_weights)

        # Create a dataloader, optimizer, and criterion

        train_dl = DataLoader(
            train_ds, batch_size=tp["batch_size"], shuffle=True, num_workers=3
        )
        validation_dl = DataLoader(
            validation_ds, batch_size=tp["batch_size"], shuffle=True, num_workers=3
        )

        print(f"Training: {len(train_dl)} batches of size {tp['batch_size']}")
        print(f"Validation: {len(validation_dl)} batches")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, whole_net.parameters()),
            lr=tp["learning_rate"],
        )
        criterion = nn.BCEWithLogitsLoss(reduction="sum")

        # For tracking
        big_tick = perf_counter()
        n_training_obs = 0

        # TKTK package this into functions

        for epoch in range(tp["n_epochs"]):

            print(f"Starting epoch {epoch+1 :2} ------")

            # Training

            whole_net.train()

            tick = perf_counter()
            running_loss = 0.0
            running_n = 0
            running_correct = 0
            running_available_history = 0

            for i, batch in enumerate(train_dl):

                optimizer.zero_grad()

                # Get a batch
                mx, sx1, mask1, sx2, mask2, y = batch

                mx = net.to_(mx, device)
                sx1 = net.to_(sx1, device)
                mask1 = mask1.to(device)
                sx2 = net.to_(sx2, device)
                mask2 = mask2.to(device)
                y = y.to(device)

                y_hat = whole_net(mx, sx1, mask1, sx2, mask2)
                labels = y_hat > 0
                correct = (labels == y).sum()

                loss = criterion(y_hat, y)

                loss.backward()
                optimizer.step()

                running_available_history += mask1.sum().item() + mask2.sum().item()

                n_training_obs += y_hat.shape[0]
                running_correct += correct.item()
                running_loss += loss.item()
                running_n += y_hat.shape[0]

                # mlflow logging
                if i % 10 == 9:
                    mlflow.log_metrics(
                        {
                            "train_loss": running_loss / running_n,
                            "train_accuracy": running_correct / running_n,
                        },
                        step=n_training_obs,
                    )

                # Print
                if i % 100 == 99:
                    print(
                        f"Epoch {epoch + 1}, Batch {i+1 :4}: {running_loss / running_n :.3f} | ",
                        f"Accuracy: {running_correct / running_n :.3f} | ",
                        f"{running_n / (perf_counter() - tick) :6.0f} obs/sec | ",
                        f"{running_available_history / (2 * running_n) :.2f} average history | "
                        f"{perf_counter() - big_tick :.2f} s",
                    )
                    running_available_history = 0
                    running_loss = 0.0
                    running_n = 0
                    running_correct = 0
                    tick = perf_counter()

            # Validation

            whole_net.eval()

            with torch.no_grad():

                tick = perf_counter()
                valid_loss = 0.0
                valid_n = 0
                valid_correct = 0

                for i, batch in enumerate(validation_dl):

                    # Get a batch
                    mx, sx1, mask1, sx2, mask2, y = batch

                    mx = net.to_(mx, device)
                    sx1 = net.to_(sx1, device)
                    mask1 = mask1.to(device)
                    sx2 = net.to_(sx2, device)
                    mask2 = mask2.to(device)
                    y = y.to(device)

                    y_hat = whole_net(mx, sx1, mask1, sx2, mask2)
                    labels = y_hat > 0
                    correct = (labels == y).sum()

                    loss = criterion(y_hat, y)

                    valid_correct += correct.item()
                    valid_loss += loss.item()
                    valid_n += y_hat.shape[0]

                print(
                    f"Epoch {epoch + 1} validation loss: {valid_loss / valid_n :.3f} | ",
                    f"Accuracy: {valid_correct / valid_n :.3f} | ",
                    f"{valid_n / (perf_counter() - tick) :6.0f} obs/sec | ",
                    f"{perf_counter() - big_tick :.2f} s",
                )

                # mlflow logging
                mlflow.log_metrics(
                    {
                        "validation_loss": valid_loss / valid_n,
                        "validation_accuracy": valid_correct / valid_n,
                    },
                    step=n_training_obs,
                )

    with open("./model/test_model_config.pkl", "wb") as f:
        pickle.dump(tp, f)
    torch.save(whole_net.state_dict(), "./model/test_model.pt")