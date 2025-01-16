import torch
import torchvision
import torchvision.transforms as transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

class NoisyDataset:
    def __init__(self, device, signal_fn, signal_strength, signal_period, total_steps):
        self.step = 0

        if signal_fn == 'const':
            self.signal_fn = lambda t: signal_strength
        elif signal_fn == 'cos':
            signal_period = torch.tensor(signal_period).to(device)
            self.signal_fn = lambda t: signal_strength / 2 * (1 + torch.cos(2 * torch.pi * t / signal_period))
        elif signal_fn == 'sin':
            signal_period = torch.tensor(signal_period).to(device)
            self.signal_fn = lambda t: signal_strength / 2 * (1 + -torch.cos(2 * torch.pi * t / signal_period))
        elif signal_fn == 'step':
            self.signal_fn = lambda t: signal_strength if t < total_steps // 3 else signal_strength / 2 if t < 2 * total_steps // 3 else 0.0
        elif signal_fn == 'step_dec':
            self.signal_fn = lambda t: 0.0 if t < total_steps // 3 else signal_strength / 2 if t < 2 * total_steps // 3 else signal_strength
        else:
            self.signal_fn = signal_fn

class CIFAR10Dataset(NoisyDataset):
    def __init__(
            self, batch_size, train=True, device="cpu", root="./data",
            signal_fn='const', signal_strength=1.0, signal_period=1000, total_steps=1000
    ):
        super().__init__(device, signal_fn, signal_strength, signal_period, total_steps)

        self.batch_size = batch_size
        self.device = device

        self.transform = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.X = (torch.tensor(dataset.data).float() / 255).permute(0, 3, 1, 2)
        self.X = self.transform(self.X).view(self.X.shape[0], -1).to(device)
        self.Y = torch.tensor(dataset.targets).to(device)

    def __iter__(self):
        while True:
            idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            X = self.X[idx]
            y = self.Y[idx]

            noise = torch.randn(X.shape).to(self.device)
            signal = self.signal_fn(self.step)
            self.step += 1
            X = X * signal + noise * (1 - signal)

            yield X, y

class SyntheticNormalDataset(NoisyDataset):
    def __init__(
            self, dataset_size, batch_size, width, device="cpu", resample=True,
            signal_fn='const', signal_strength=1.0, signal_period=1000, total_steps=1000
    ):
        super().__init__(device, signal_fn, signal_strength, signal_period, total_steps)

        self.batch_size = batch_size
        self.device = device
        self.width = width
        self.dataset_size = dataset_size
        self.resample = resample

        # Generate synthetic data and labels from N(0, 1)
        self.X = torch.randn(self.dataset_size, width).to(device)

        # Generate weights for a linear relationship
        true_weights = torch.randn(width, 1).to(device)
        self.Y = self.X @ true_weights

    def __iter__(self):
        while True:
            if self.resample:
                idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            else:
                idx = torch.randperm(self.X.shape[0])[:self.batch_size]
                if self.batch_size == self.dataset_size:
                    idx = torch.arange(0, self.X.shape[0])

            X = self.X[idx]
            y = self.Y[idx]

            noise = torch.randn(X.shape).to(self.device)
            signal = self.signal_fn(self.step)
            self.step += 1
            X = X * signal + noise * (1 - signal)

            yield X, y


if __name__ == "__main__":
    ds = CIFAR10Dataset(4, True)
    for X, y in ds:
        print(X.shape, y.shape)
        break