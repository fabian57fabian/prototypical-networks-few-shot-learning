import torch.nn as nn


class PrototypicalNetwork(nn.Module):

    def _conv_layer(self, input, output, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size, padding=padding),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def __init__(self, input_units=3, hidden_units=64, output_units=64, kernel_size=3):
        super(PrototypicalNetwork, self).__init__()
        self.net = nn.Sequential(
            self._conv_layer(input_units, hidden_units, kernel_size, 1),
            self._conv_layer(hidden_units, hidden_units, kernel_size, 1),
            self._conv_layer(hidden_units, hidden_units, kernel_size, 1),
            self._conv_layer(hidden_units, output_units, kernel_size, 1),
        )

    def forward(self, x):
        y = self.net(x)
        return y.view(y.size(0), -1)

