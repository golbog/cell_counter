import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops

from . import modules as mdls


class ResUNetPP(nn.Module):
    def __init__(self, num_input_channels, filters=(16, 32, 64, 128, 256)):
        super().__init__()
        self.num_channels = num_input_channels
        self.filters = filters
        self.input_res = nn.Sequential(
            nn.Conv2d(num_input_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.SiLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(num_input_channels, filters[0], kernel_size=3, padding=1)

        # encoder
        self.se1 = ops.SqueezeExcitation(filters[0], 16)
        self.rb1 = mdls.ResidualUnit(filters[0], filters[1], stride=2)

        self.se2 = ops.SqueezeExcitation(filters[1], 16)
        self.rb2 = mdls.ResidualUnit(filters[1], filters[2], stride=2)

        self.se3 = ops.SqueezeExcitation(filters[2], 16)
        self.rb3 = mdls.ResidualUnit(filters[2], filters[3], stride=2)

        # bridge
        self.aspp_bridge = mdls.ASPPBlock(filters[3], filters[4])

        # decoder
        self.attn1 = mdls.ABlock(filters[2], filters[4], filters[4])
        self.upsample1 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.up_rb1 = mdls.ResidualUnit(filters[4] + filters[2], filters[3])

        self.attn2 = mdls.ABlock(filters[1], filters[3], filters[3])
        self.upsample2 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.up_rb2 = mdls.ResidualUnit(filters[3] + filters[1], filters[2])

        self.attn3 = mdls.ABlock(filters[0], filters[2], filters[2])
        self.upsample3 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.up_rb3 = mdls.ResidualUnit(filters[2] + filters[0], filters[1])

        self.aspp_out = mdls.ASPPBlock(filters[1], filters[0])
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def forward(self, x):
        x1 = self.input_res(x) + self.input_skip(x)

        x2 = self.se1(x1)
        x2 = self.rb1(x2)

        x3 = self.se2(x2)
        x3 = self.rb2(x3)

        x4 = self.se3(x3)
        x4 = self.rb3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_rb1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_rb2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_rb3(x8)

        x9 = self.aspp_out(x8)
        res = self.output_layer(x9)

        return res

    def get_size_mult(self):
        return self.filters[0]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    r = ResUNetPP(3)
    img = torch.rand((1, 3, 1024, 1024))
    print(r(img).shape)
