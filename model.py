import torch
from torch import nn


class SpotterLayer(nn.Module):
    ...


class SVDF(SpotterLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = kernel

        self.svdf = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel),
            nn.Conv1d(out_channels, out_channels, 1, groups=out_channels)
        )

        self.conv_grad_mask = None
        self.depthwise_conv_grad_mask = None

    def forward(self, x):
        return self.svdf(x)


class Conv1d(SpotterLayer):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv = nn.Conv1d(*args, **kwargs)
        self.conv_grad_mask = None

    def forward(self, x):
        return self.conv(x)


class BatchNorm1d(SpotterLayer):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(*args, **kwargs)
        self.grad_mask = None

    def forward(self, x):
        return self.batchnorm(x)


class ReLU(SpotterLayer):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class Spotter(SpotterLayer):
    def __init__(self, n_mels: int, neurons: int, btneck: int, n_classes: int):
        super().__init__()

        self.proj = nn.Sequential(
            BatchNorm1d(n_mels),
            Conv1d(n_mels, neurons, 1),
        )

        self.encoder = nn.Sequential(
            BatchNorm1d(neurons),
            ReLU(),
            Conv1d(neurons, btneck, 1),
            SVDF(btneck, neurons, 11),
            BatchNorm1d(neurons),
            ReLU(),
            Conv1d(neurons, btneck, 1),
            SVDF(btneck, neurons, 11),
        )

        self.classifier = Conv1d(neurons, n_classes, 1)

    def forward(self, x):
        projected = self.proj(x.clamp(-1e5, 1e5))
        encoding = self.encoder(projected)
        pooled = torch.mean(encoding, dim=-1, keepdim=True)

        return self.classifier(pooled)
