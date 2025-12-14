import torch
import torch.nn as nn

from . import modules as mdls


class CResUnet(nn.Module):
    def __init__(self, num_channels=3, filters=(32, 64, 128, 256)):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        self.deconvs = torch.nn.ModuleList()
        self.num_channels = num_channels
        self.filters = filters

        # input
        self.input_conv = nn.Conv2d(num_channels, 1, kernel_size=1)

        # down
        self.blocks.append(mdls.ResidualUnit(1, filters[0], 3))
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blocks.append(mdls.ResidualUnit(filters[0], filters[1], 3))
        self.blocks.append(mdls.ResidualUnit(filters[1], filters[2], 3))

        # bridge
        self.blocks.append(mdls.ResidualUnit(filters[2], filters[3], 3))
        # self.blocks.append(ResidualUnit(filters[3], filters[3], 3))
        # TODO: in the original model, blocks below this have skip but without convolution
        self.blocks.append(mdls.ResidualUnit(filters[3], filters[3], 3))

        # up
        self.deconvs.append(nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2))
        self.blocks.append(mdls.ResidualUnit(filters[3], filters[2], 3))
        self.deconvs.append(nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2))
        self.blocks.append(mdls.ResidualUnit(filters[2], filters[1], 3))
        self.deconvs.append(nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2))
        self.blocks.append(mdls.ResidualUnit(filters[1], filters[0], 3))

        # output
        self.output = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def forward(self, x):
        x = self.input_conv(x)
        p1 = self.blocks[0](x)

        x = self.max_pooling(p1)
        p2 = self.blocks[1](x)

        x = self.max_pooling(p2)
        p3 = self.blocks[2](x)

        # bridge
        x = self.max_pooling(p3)
        x = self.blocks[3](x)
        x = self.blocks[4](x)
        x = self.deconvs[0](x)

        # up
        c_x = torch.concat((x, p3), dim=1)
        x = self.blocks[5](c_x)
        x = self.deconvs[1](x)
        c_x = torch.concat((x, p2), dim=1)
        x = self.blocks[6](c_x)
        x = self.deconvs[2](x)
        c_x = torch.concat((x, p1), dim=1)
        x = self.blocks[7](c_x)
        x = self.output(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_size_mult(self):
        return self.filters[0]
