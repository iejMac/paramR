import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.layer_multiplier = 1.0

    def forward(self, x):
        x = self.lin(x)
        x = x * self.layer_multiplier
        x = F.relu(x)
        # x = F.tanh(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims, bias=False):
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1
        self.layers = nn.ModuleList([MLPLayer(dims[i], dims[i+1], bias=bias) for i in range(self.n_layers)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


if __name__ == "__main__":
    mod = MLP([2, 3])
    x = torch.randn(2)
    y = mod(x)
    print(y.shape)