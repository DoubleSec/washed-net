import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, size: int):

        super().__init__()

        self.size = size
        # TKTK Is this the right initialization?
        self.weight = nn.Parameter(
            torch.normal(torch.zeros([1, self.size]), 1, requires_grad=True)
        )
        self.bias = nn.Parameter(
            torch.normal(torch.zeros([1, self.size]), 1, requires_grad=True)
        )

    def forward(self, x):

        linear_term = self.weight[:, 0] * x + self.bias[:, 0]
        periodic_terms = torch.sin(self.weight[:, 1:] * x + self.bias[:, 1:])

        return torch.cat([linear_term, periodic_terms], dim=1)


if __name__ == "__main__":

    # Test time2vec
    x = torch.normal(torch.zeros([32, 1]), 1)
    l = Time2Vec(8)
    l(x)