import torch
from torch import nn

from .data import DataInterface


class Time2Vec(nn.Module):
    def __init__(self, size: int):

        super().__init__()

        self.size = size
        # TKTK Is this the right initialization?
        self.weight = nn.Parameter(
            torch.normal(torch.zeros([1, self.size]), 1.0)
        )
        self.bias = nn.Parameter(
            torch.normal(torch.zeros([1, self.size]), 1.0)
        )

    def forward(self, x):

        linear_term = self.weight[:, 0] * x + self.bias[:, 0]
        periodic_terms = torch.sin(self.weight[:, 1:] * x + self.bias[:, 1:])

        return torch.cat([linear_term, periodic_terms], dim=1)


class TabularInputLayer(nn.Module):
    def __init__(
        self,
        interface: DataInterface,
        col_encoding_size: int,
        embedding_size: int,
    ):

        super().__init__()

        self.interface = interface
        self.embedding_size = embedding_size
        self.col_encoding_size = col_encoding_size

        # Initialize the embedding layer for each column
        self.embedding = nn.ModuleDict()
        for k, v in self.interface.type_map.items():

            if v[0] == "numeric":
                self.embedding[k] = nn.Linear(v[2].shape[0], embedding_size)

            elif v[0] == "categorical":
                self.embedding[k] = nn.Embedding(len(v[1]), embedding_size)

            else:
                self.embedding[k] = Time2Vec(embedding_size)

        # Initialize the column encoding
        self.col_encoding = nn.Parameter(torch.normal(torch.zeros([len(self.interface), self.col_encoding_size]), 1.0))
        # self.col_encoding = nn.Embedding(len(self.interface), self.col_encoding_size)

    def forward(self, x: dict):

        # This will break if the batch size is 1
        # n x e x s
        x = torch.stack([self.embedding[k](v).squeeze() for k, v in x.items()], dim=-1)
        
        # permute tensor to n x s x e
        x = torch.permute(x, [0, 2, 1])

        # s x e
        exp_col_encoding = self.col_encoding.unsqueeze(0).expand([x.shape[0], -1, -1])

        x = torch.cat([x, exp_col_encoding], dim=-1)

        return x



if __name__ == "__main__":

    # Test time2vec
    x = torch.normal(torch.zeros([32, 1]), 1)
    l = Time2Vec(8)
    l(x)
