import torch
from torch import nn
from torch.utils.data import Dataset

from .data import DataInterface
from .net import Time2Vec, Squeezer


class PretrainingDataset(Dataset):
    def __init__(self, input_data: dict):

        super().__init__()

        self.data = input_data

    def __len__(self):
        return self.data["p2_hand"].shape[0]

    def __getitem__(self, idx):

        x = {k: v[idx, :] for k, v in self.data.items()}

        return x


class CutMix(nn.Module):

    """CutMix a single tensor"""

    def __init__(self, p: float):

        super().__init__()

        # p is the likelihood of switching
        self.p = torch.tensor(p)

    def forward(self, x):

        x_type = x.dtype

        # This is not quite right; it can map observations to themselves.
        mix_pairings = torch.randperm(x.shape[0])
        mask = torch.bernoulli(self.p.expand(x.shape[0], 1)).to(x.device)

        # This relies on x being n x k; if x is one dimensional it won't work.
        mixed_x = x * (1 - mask) + x[mix_pairings] * mask
        return mixed_x.type(x_type)


class PretrainingNet(nn.Module):
    """This is a class for pretraining."""

    def __init__(
        self,
        interface: DataInterface,
        embedding_size: int,
    ):

        super().__init__()

        self.interface = interface
        self.embedding_size = embedding_size

        # Initialize the embedding layer for each column
        self.embedding = nn.ModuleDict()
        for k, v in self.interface.type_map.items():

            if v[0] == "numeric":
                self.embedding[k] = nn.Linear(v[2].shape[0] - 1, embedding_size)

            elif v[0] == "categorical":
                self.embedding[k] = nn.Sequential(
                    Squeezer(-1), nn.Embedding(len(v[1]), embedding_size)
                )

            else:  # Time layers
                self.embedding[k] = nn.Sequential(
                    Time2Vec(embedding_size),
                    nn.LayerNorm(embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size, embedding_size),
                )

        self.projection_layers = nn.Sequential(
            nn.Linear(self.embedding_size * len(self.embedding), self.embedding_size),
            nn.LayerNorm([self.embedding_size]),
            nn.ReLU(),
        )

    def forward(self, x: dict):

        # n x e
        x = torch.cat([self.embedding[k](v) for k, v in x.items()], dim=-1)

        x = self.projection_layers(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc(x)
        x = self.relu(x)
        return x
