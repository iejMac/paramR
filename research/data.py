import torch
import torchvision
import torchvision.transforms as transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

class NoisyDataset:
    def __init__(self, device, signal_fn, signal_strength, signal_range, signal_period, total_steps):
        self.step = 0

        if signal_fn == 'const':
            # Always return the same constant signal_strength.
            self.signal_fn = lambda t: signal_strength
        elif signal_fn == 'cos':
            # Oscillates between (signal_strength - signal_range/2) and (signal_strength + signal_range/2)
            signal_period = torch.tensor(signal_period).to(device)
            self.signal_fn = lambda t: signal_strength + (signal_range / 2) * torch.cos(2 * torch.pi * t / signal_period)
        elif signal_fn == 'sin':
            # Similarly for sine oscillation.
            signal_period = torch.tensor(signal_period).to(device)
            self.signal_fn = lambda t: signal_strength + (signal_range / 2) * torch.sin(2 * torch.pi * t / signal_period)
        elif signal_fn == 'step':
            # A simple step function.
            self.signal_fn = lambda t: signal_strength if t < total_steps // 3 else signal_strength / 2 if t < 2 * total_steps // 3 else 0.0
        elif signal_fn == 'step_dec':
            # Another variant of a step function.
            self.signal_fn = lambda t: 0.0 if t < total_steps // 3 else signal_strength / 2 if t < 2 * total_steps // 3 else signal_strength
        else:
            # If the user provides a custom function, use it.
            self.signal_fn = signal_fn

class CIFAR10Dataset(NoisyDataset):
    def __init__(
            self, batch_size, train=True, device="cpu", root="./data",
            signal_fn='const', signal_strength=1.0, signal_range=1.0, signal_period=1000, total_steps=1000
    ):
        # Pass the new parameter signal_range to the parent class.
        super().__init__(device, signal_fn, signal_strength, signal_range, signal_period, total_steps)
        self.batch_size = batch_size
        self.device = device
        self.type = "classification"

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
            # Get the current signal value.
            signal = self.signal_fn(self.step)
            self.step += 1
            # Blend the original data with noise based on the signal.
            X = X * signal + noise * (1 - signal)

            yield X, y

class SyntheticNormalDataset(NoisyDataset):
    def __init__(
            self, dataset_size, batch_size, width, device="cpu", resample=True,
            signal_fn='const', signal_strength=1.0, signal_range=1.0, signal_period=1000, total_steps=1000
    ):
        super().__init__(device, signal_fn, signal_strength, signal_range, signal_period, total_steps)
        self.batch_size = batch_size
        self.device = device
        self.width = width
        self.dataset_size = dataset_size
        self.resample = resample
        self.type = "regression"

        # Generate synthetic data from a standard normal distribution.
        self.X = torch.randn(self.dataset_size, width).to(device)

        # Create a linear relationship for labels.
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
    # Example usage: a cosine signal that oscillates between 1.0 - 0.1 and 1.0 + 0.1
    ds = CIFAR10Dataset(
        batch_size=4,
        train=True,
        device="cpu",
        signal_fn='cos',
        signal_strength=0.6,   # center of oscillation
        signal_range=0.2,      # total oscillation range (i.e., ±0.1)
        signal_period=100,
        total_steps=1000
    )
    for X, y in ds:
        print(X.shape, y.shape)
        break