"""SegNet baseline adapted from vinceecws/SegNet_PyTorch.

The five-stage VGG-style encoder and index-based MaxUnpool decoder follow the
upstream implementation. The benchmark-facing forward method returns logits
instead of applying softmax, which is the expected output for CrossEntropyLoss
and keeps the interface consistent with the other segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        bn_momentum=0.5,
        apply_softmax=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.apply_softmax = apply_softmax

        self.max_en = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.max_de = nn.MaxUnpool2d(2, stride=2)

        # Encoder: VGG-style stages 1-2 use two convolutions; stages 3-5 use
        # three convolutions.
        self.conv_en11 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn_en11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv_en12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_en12 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.conv_en21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_en21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.conv_en22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_en22 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.conv_en31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_en31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv_en32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_en32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv_en33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_en33 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.conv_en41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_en41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_en42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_en42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_en43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_en43 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.conv_en51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_en51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_en52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_en52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_en53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_en53 = nn.BatchNorm2d(512, momentum=bn_momentum)

        # Decoder mirrors the encoder channel widths.
        self.conv_de53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_de53 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_de52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_de52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_de51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_de51 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.conv_de43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_de43 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_de42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_de42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv_de41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_de41 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.conv_de33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_de33 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv_de32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_de32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv_de31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_de31 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.conv_de22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_de22 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.conv_de21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_de21 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.conv_de12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_de12 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv_de11 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.bn_de11 = nn.BatchNorm2d(n_classes, momentum=bn_momentum)

    @staticmethod
    def _relu_bn(conv, bn, x):
        return F.relu(bn(conv(x)))

    def forward(self, x):
        x = self._relu_bn(self.conv_en11, self.bn_en11, x)
        x = self._relu_bn(self.conv_en12, self.bn_en12, x)
        size0 = x.size()
        x, ind1 = self.max_en(x)
        size1 = x.size()

        x = self._relu_bn(self.conv_en21, self.bn_en21, x)
        x = self._relu_bn(self.conv_en22, self.bn_en22, x)
        x, ind2 = self.max_en(x)
        size2 = x.size()

        x = self._relu_bn(self.conv_en31, self.bn_en31, x)
        x = self._relu_bn(self.conv_en32, self.bn_en32, x)
        x = self._relu_bn(self.conv_en33, self.bn_en33, x)
        x, ind3 = self.max_en(x)
        size3 = x.size()

        x = self._relu_bn(self.conv_en41, self.bn_en41, x)
        x = self._relu_bn(self.conv_en42, self.bn_en42, x)
        x = self._relu_bn(self.conv_en43, self.bn_en43, x)
        x, ind4 = self.max_en(x)
        size4 = x.size()

        x = self._relu_bn(self.conv_en51, self.bn_en51, x)
        x = self._relu_bn(self.conv_en52, self.bn_en52, x)
        x = self._relu_bn(self.conv_en53, self.bn_en53, x)
        x, ind5 = self.max_en(x)

        x = self.max_de(x, ind5, output_size=size4)
        x = self._relu_bn(self.conv_de53, self.bn_de53, x)
        x = self._relu_bn(self.conv_de52, self.bn_de52, x)
        x = self._relu_bn(self.conv_de51, self.bn_de51, x)

        x = self.max_de(x, ind4, output_size=size3)
        x = self._relu_bn(self.conv_de43, self.bn_de43, x)
        x = self._relu_bn(self.conv_de42, self.bn_de42, x)
        x = self._relu_bn(self.conv_de41, self.bn_de41, x)

        x = self.max_de(x, ind3, output_size=size2)
        x = self._relu_bn(self.conv_de33, self.bn_de33, x)
        x = self._relu_bn(self.conv_de32, self.bn_de32, x)
        x = self._relu_bn(self.conv_de31, self.bn_de31, x)

        x = self.max_de(x, ind2, output_size=size1)
        x = self._relu_bn(self.conv_de22, self.bn_de22, x)
        x = self._relu_bn(self.conv_de21, self.bn_de21, x)

        x = self.max_de(x, ind1, output_size=size0)
        x = self._relu_bn(self.conv_de12, self.bn_de12, x)
        logits = self.conv_de11(x)
        return F.softmax(logits, dim=1) if self.apply_softmax else logits
