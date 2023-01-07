import torch
from torch import nn
from torch.nn import functional as F

from .data import DataInterface


# Move every tensor in a dict to a device
def to_(x: dict, device):

    return {k: v.to(device) for k, v in x.items()}


class Time2Vec(nn.Module):
    def __init__(self, size: int):

        super().__init__()

        self.size = size
        # TKTK Is this the right initialization?
        self.weight = nn.Parameter(torch.normal(torch.zeros([1, self.size]), 1.0))
        self.bias = nn.Parameter(torch.normal(torch.zeros([1, self.size]), 1.0))

    def forward(self, x):

        linear_term = self.weight[:, 0] * x + self.bias[:, 0]
        periodic_terms = torch.sin(self.weight[:, 1:] * x + self.bias[:, 1:])

        return torch.cat([linear_term, periodic_terms], dim=-1)


class TabularInputLayer(nn.Module):
    def __init__(
        self,
        interface: DataInterface,
        col_encoding_size: int,
        embedding_size: int,
        append_cls: bool = True,
    ):

        super().__init__()

        self.interface = interface
        self.embedding_size = embedding_size
        self.col_encoding_size = col_encoding_size
        self.append_cls = append_cls

        # Initialize the embedding layer for each column
        self.embedding = nn.ModuleDict()
        for k, v in self.interface.type_map.items():

            if v[0] == "numeric":
                self.embedding[k] = nn.Linear(v[2].shape[0], embedding_size)

            elif v[0] == "categorical":
                self.embedding[k] = nn.Embedding(len(v[1]), embedding_size)

            else:
                # I don't know if this is justified, but the idea here is to
                # additionally transform the time2vec embeddings before we
                # provide them to the transformer, analogous to the numeric
                # embeddings.
                self.embedding[k] = nn.Sequential(
                    Time2Vec(embedding_size), nn.Linear(embedding_size, embedding_size)
                )

        # Initialize the column encoding
        self.col_encoding = nn.Parameter(
            torch.normal(
                torch.zeros([len(self.interface), self.col_encoding_size]), 1.0
            )
        )

        # Initialize the <cls> embedding, if needed
        if self.append_cls:
            self.cls_embedding = nn.Parameter(
                torch.normal(
                    torch.zeros([1, 1, self.embedding_size + self.col_encoding_size]),
                    1.0,
                )
            )

    def forward(self, x: dict):

        # This will break if the batch size is 1
        # n x e x s
        x = torch.stack([self.embedding[k](v).squeeze() for k, v in x.items()], dim=-1)

        # permute tensor to n x s x e
        x = torch.permute(x, [0, 2, 1])

        # Expand the column encodings so we can concatenate them (s x e -> n x s x e)
        exp_col_encoding = self.col_encoding.unsqueeze(0).expand([x.shape[0], -1, -1])

        # Attach the column encodings to the encoded data.
        # n x s x e
        x = torch.cat([x, exp_col_encoding], dim=-1)

        if self.append_cls:
            expanded_cls = self.cls_embedding.expand([x.shape[0], -1, -1])
            x = torch.cat([x, expanded_cls], dim=1)

        return x


class SequentialInputLayer(nn.Module):
    """This is similar to the TabularInputLayer except:
    - There are sequence encodings instead of column encodings, although they're
      functionally almost identical.
    - There's a padding embedding for padding positions, and it takes a padding mask
    as an input.
    - The embeddings from the features are simply concatenated and then projected
    into the embedding space for the transformer inputs, for each sequential
    observation.
    It's an open question whether I need more activations, BatchNorm, etc."""

    def __init__(
        self,
        interface: DataInterface,
        sequence_encoding_size,
        embedding_size: int,
    ):

        super().__init__()

        self.interface = interface
        self.sequence_encoding_size = sequence_encoding_size
        self.embedding_size = embedding_size

        # Initialize the embedding layer for each column
        self.embedding = nn.ModuleDict()
        for k, v in self.interface.type_map.items():

            if v[0] == "numeric":
                self.embedding[k] = nn.Linear(v[2].shape[0], embedding_size)

            elif v[0] == "categorical":
                self.embedding[k] = nn.Sequential(
                    Squeezer(-1), nn.Embedding(len(v[1]), embedding_size)
                )

            else:
                # I don't know if this is justified, but the idea here is to
                # additionally transform the time2vec embeddings before we
                # provide them to the transformer, analogous to the numeric
                # embeddings.
                self.embedding[k] = nn.Sequential(
                    Time2Vec(embedding_size), nn.Linear(embedding_size, embedding_size)
                )

        self.projection_layer = nn.Linear(
            self.embedding_size * len(self.embedding), self.embedding_size
        )

        # Initialize the column encoding
        self.sequence_encoding = nn.Parameter(
            torch.normal(torch.zeros(self.sequence_encoding_size), 1.0)
        )

        self.padding_encoding = nn.Parameter(
            torch.normal(torch.zeros(1, self.sequence_encoding_size[1]), 1.0)
        )

    def forward(self, x: dict, mask: torch.Tensor):

        # This will break if the batch size is 1
        # n x s x e
        x = torch.cat([self.embedding[k](v) for k, v in x.items()], dim=-1)

        x = F.relu(x)
        x = self.projection_layer(x)

        # Expand the sequence encodings so we can concatenate them (s x e -> n x s x e)
        exp_seq_encoding = self.sequence_encoding.unsqueeze(0).expand(
            [x.shape[0], -1, -1]
        )

        # black has no idea how to format this, huh
        final_seq_encoding = exp_seq_encoding * mask.unsqueeze(
            -1
        ) + self.padding_encoding.unsqueeze(0) * ~mask.unsqueeze(-1)

        # Attach the column encodings to the encoded data.
        # n x s x e
        x = torch.cat([x, final_seq_encoding], dim=-1)

        return x


class FusionNet(nn.Module):
    """This class is a replacement for the nn.Sequential in the first version, because
    we need to do very slightly more work to get everything put together."""

    def __init__(
        self,
        table_input_layer,
        p1_sequence_input_layer,
        p2_sequence_input_layer,
        transformer,
        output_layer,
    ):

        super().__init__()

        self.table_input_layer = table_input_layer
        self.p1_sequence_input_layer = p1_sequence_input_layer
        self.p2_sequence_input_layer = p2_sequence_input_layer
        self.transformer = transformer
        self.output_layer = output_layer

    def forward(self, tx, sx1, mask1, sx2, mask2):

        tx = self.table_input_layer(tx)
        sx1 = self.p1_sequence_input_layer(sx1, mask1)
        sx2 = self.p2_sequence_input_layer(sx2, mask2)

        # Choosing this order so it's easier to find the CLS embedding later.
        x = torch.cat([sx1, sx2, tx], dim=1)
        x = self.transformer(x)
        x = self.output_layer(x[:, -1, :])

        return x


# fc layers for output
class OutputLayers(nn.Module):
    def __init__(self, d_model, n_hidden, output_size):
        """d_model should be the same as the transformer encoder"""

        super().__init__()
        self.d_model = d_model
        self.output_size = output_size
        self.n_hidden = n_hidden

        self.layers = nn.Sequential()

        # Add the hidden layers
        for i in range(n_hidden):
            self.layers.append(nn.Linear(d_model, d_model))
            self.layers.append(nn.BatchNorm1d(d_model))
            self.layers.append(nn.ReLU())

        # Add the output layer (no output activation)
        self.layers.append(nn.Linear(d_model, output_size))

    def forward(self, x):

        # forward() from the sequential module does what we want
        return self.layers(x)


class Squeezer(nn.Module):
    def __init__(self, dim):

        super().__init__()

        self.dim = dim

    def forward(self, x):

        return x.squeeze(dim=self.dim)


class SelectCLSEncoding(nn.Module):
    """Trivial class so I can use nn.Sequential. Nothing learnable in here"""

    def __init__(self):

        super().__init__()

    def forward(self, x):
        # Get the last value in this dimension (the CLS encoding)
        return x[:, -1, :]


if __name__ == "__main__":

    pass
