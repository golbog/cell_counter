import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(in_channels, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(out_channels, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class ABlock(nn.Module):
    def __init__(self, input_dim, decoder_dim, output_dim, kernel_size=3):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size, padding=kernel_size // 2),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, output_dim, kernel_size, padding=kernel_size // 2),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class ASPPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, rates=(2, 4, 8), kernel_size=3):
        super().__init__()

        self.aspp_blocks = torch.nn.ModuleList()

        for rate in rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_dim, output_dim, kernel_size, stride=1, padding=rate, dilation=rate
                    ),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(output_dim),
                )
            )

        self.output = nn.Conv2d(len(rates) * output_dim, output_dim, 1)

    def forward(self, x):
        x = torch.cat([aspp_block(x) for aspp_block in self.aspp_blocks], dim=1)
        return self.output(x)
