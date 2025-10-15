import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.layer_multiplier = 1.0
        self.activation_fn = activation or nn.Identity()

    def forward(self, x):
        x = self.lin(x)
        x = x * self.layer_multiplier
        x = self.activation_fn(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.ln = nn.LayerNorm(dim, bias=bias)
        self.layer_multiplier = 1.0

    def forward(self, x):
        x = self.ln(x)
        x = x * self.layer_multiplier
        return x


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.params = nn.Parameter(torch.randn(1, in_dim, out_dim))
        self.layer_multiplier = 1.0

    def forward(self):
        return self.params * self.layer_multiplier


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.n_layers = len(dims) - 1

        layers = [MLPLayer(dims[i], dims[i + 1]) for i in range(self.n_layers - 1)]
        layers.append(MLPLayer(dims[-2], dims[-1], activation=None))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForwardBlock, self).__init__()
        self.dense1 = MLPLayer(dim, hidden_dim, activation=nn.GELU())
        self.dense2 = MLPLayer(hidden_dim, dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.k = MLPLayer(dim, dim)
        self.q = MLPLayer(dim, dim)
        self.v = MLPLayer(dim, dim)
        self.scale = 1 / dim  # muP scaling

    def forward(self, x):
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attn = AttentionBlock(dim)
        self.ffn = FeedForwardBlock(dim, hidden_dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, hidden_dim) for _ in range(n_layers)])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super(Patchify, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, -1, self.patch_size * self.patch_size * C)
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, hidden_dim, n_layers, n_classes):
        super(ViT, self).__init__()
        self.patchify = Patchify(patch_size)
        self.embed = MLPLayer(patch_size * patch_size * 3, dim)
        self.cls_token = Embedding(1, dim)
        self.pos_embed = Embedding((image_size // patch_size) ** 2 + 1, dim)
        self.transformer = Transformer(dim, hidden_dim, n_layers)
        self.readout = MLPLayer(dim, n_classes)

    def forward(self, x):
        b, *_ = x.shape

        x = self.patchify(x)
        x = self.embed(x)

        cls_tokens = self.cls_token().expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed()

        x = self.transformer(x)
        x = self.readout(x[:, 0])

        return x


if __name__ == "__main__":
    mod = MLP([2, 3])
    x = torch.randn(2)
    y = mod(x)
    print(y.shape)