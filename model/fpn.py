import torch
import torch.nn as nn
import torch.nn.functional as F
from . import modules as mdls


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) for semantic segmentation / cell counting.
    This architecture uses a ResNet-like backbone and a top-down pathway with lateral connections
    to build a feature pyramid. The features from all levels are then upsampled and concatenated
    to form the final prediction.
    """
    def __init__(self, num_channels=3, filters=(16, 32, 64, 128, 256)):
        super().__init__()
        self.filters = filters
        self.num_channels = num_channels

        # encoder (bottom-up)
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, filters[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # encoder stages
        self.enc1 = mdls.ResidualUnit(filters[0], filters[1], stride=2) # Stride 2 -> 1/2 resolution
        self.enc2 = mdls.ResidualUnit(filters[1], filters[2], stride=2) # Stride 2 -> 1/4 resolution
        self.enc3 = mdls.ResidualUnit(filters[2], filters[3], stride=2) # Stride 2 -> 1/8 resolution
        self.enc4 = mdls.ResidualUnit(filters[3], filters[4], stride=2) # Stride 2 -> 1/16 resolution
        
        # fpn lateral connections
        # Project all feature maps to a common channel dimension
        fpn_dim = filters[2]
        
        self.lat4 = nn.Conv2d(filters[4], fpn_dim, 1)
        self.lat3 = nn.Conv2d(filters[3], fpn_dim, 1)
        self.lat2 = nn.Conv2d(filters[2], fpn_dim, 1)
        self.lat1 = nn.Conv2d(filters[1], fpn_dim, 1)
        
        # smooth layers (anti-aliasing after summation)
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        
        # segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        c0 = self.stem(x)      # 1/1
        c1 = self.enc1(c0)     # 1/2
        c2 = self.enc2(c1)     # 1/4
        c3 = self.enc3(c2)     # 1/8
        c4 = self.enc4(c3)     # 1/16
        
        # top-down pathway
        p4 = self.lat4(c4)
        
        # P3 = Lat3(C3) + Upsample(P4)
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # P2 = Lat2(C2) + Upsample(P3)
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        
        # P1 = Lat1(C1) + Upsample(P2)
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        
        # upsample all to input size and concatenate
        size = x.shape[2:]
        
        p4_up = F.interpolate(p4, size=size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=size, mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=size, mode='bilinear', align_corners=False)
        p1_up = F.interpolate(p1, size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([p4_up, p3_up, p2_up, p1_up], dim=1)
        
        return self.final_conv(out)

    def get_size_mult(self):
        return 32
