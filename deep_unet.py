from typing import List

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, hidden_layer=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            ConvSamePad2d(in_channels=in_channels, out_channels=hidden_layer, kernel_size=3),
            nn.ReLU(),
            ConvSamePad2d(in_channels=hidden_layer, out_channels=out_channels, kernel_size=2),
        )

    def forward(self, inputs):
        return self.layer(inputs)


class ConvSamePad2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__()

        left_top_pad = right_bottom_pad = kernel_size // 2
        if kernel_size % 2 == 0:
            right_bottom_pad -= 1

        self.layer = nn.Sequential(
            nn.ReflectionPad2d((left_top_pad, right_bottom_pad, left_top_pad, right_bottom_pad)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class DownBlock_FirstLayer(nn.Module):
    """
    This layer is the first layer of DownBlock
    that doesn't include element-wise addition with the feature map from the previous DownBlock.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32, hidden_layer: List[int] = [64, 64]):
        super().__init__()

        input_layer = [in_channels] + hidden_layer
        output_layer = hidden_layer + [out_channels]

        self.layer = nn.Sequential()
        for idx in range(len(input_layer)):
            self.layer.add_module(
                f'first_conv_{idx}', nn.Sequential(
                    nn.ReLU(),
                    ConvSamePad2d(in_channels=input_layer[idx], out_channels=output_layer[idx], kernel_size=3))
            )

        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        pooled_outputs = self.pool(outputs)
        return pooled_outputs, outputs


class DownBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, hidden_layer=64):
        super().__init__()
        self.layer = ConvLayer(in_channels, out_channels, hidden_layer)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = torch.add(outputs, inputs)
        pooled_outputs = self.pool(outputs)
        return pooled_outputs, outputs


class UpBlock_FinalLayer(nn.Module):
    """
    This layer is the final layer of UpBlock
    that doesn't include the UpSampling.
    """

    def __init__(self, in_channels=32, out_channels=32, hidden_layer=64):
        super().__init__()
        self.layer = ConvLayer(in_channels, out_channels, hidden_layer)

    def forward(self, inputs1, inputs2):
        """
        Args:
            inputs1: feature map from the previous UpBlock.
            inputs2: from the DownBlock through u-connection.
        """
        concats = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.layer(concats)
        outputs = torch.add(outputs, inputs1)
        return outputs


class UpBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, hidden_layer=64):
        super().__init__()
        self.layer = ConvLayer(in_channels, out_channels, hidden_layer)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        """
        Args:
            inputs1: feature map from the previous UpBlock.
            inputs2: from the DownBlock through u-connection.
        """
        concats = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.layer(concats)
        outputs = torch.add(outputs, inputs1)
        outputs = self.up_sample(outputs)
        return outputs


class DeepUnet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()

        # Here, the input shape of this first DownBlock is (`batch_size`, 3, 640, 640)
        self.down_conv1 = DownBlock_FirstLayer(in_channels=in_channels, out_channels=32, hidden_layer=[64, 64])
        self.down_conv2 = DownBlock(in_channels=32, out_channels=32, hidden_layer=64)
        self.down_conv3 = DownBlock(in_channels=32, out_channels=32, hidden_layer=64)
        self.down_conv4 = DownBlock(in_channels=32, out_channels=32, hidden_layer=64)
        self.down_conv5 = DownBlock(in_channels=32, out_channels=32, hidden_layer=64)
        self.down_conv6 = DownBlock(in_channels=32, out_channels=32, hidden_layer=64)
        self.conv_layer = ConvLayer(in_channels=32, out_channels=32, hidden_layer=64)

        self.up_conv1 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv2 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv3 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv4 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv5 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv6 = UpBlock(in_channels=64, out_channels=32, hidden_layer=64)
        self.up_conv7 = UpBlock_FinalLayer(in_channels=64, out_channels=32, hidden_layer=64)

        self.final_conv_layer = nn.Sequential(
            nn.ReLU(),
            ConvSamePad2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            ConvSamePad2d(in_channels=32, out_channels=n_classes, kernel_size=3)
        )

    def forward(self, inputs):
        assert inputs.size()[-2:] == (640, 640), ValueError(f"Input shape mismatch. It should be 640x640 while the input shape is {inputs.size()[-2:]}")

        outputs, u_connection_1 = self.down_conv1(inputs)
        outputs, u_connection_2 = self.down_conv2(outputs)
        outputs, u_connection_3 = self.down_conv3(outputs)
        outputs, u_connection_4 = self.down_conv4(outputs)
        outputs, u_connection_5 = self.down_conv5(outputs)
        outputs, u_connection_6 = self.down_conv6(outputs)
        outputs = self.conv_layer(outputs)

        outputs = self.up_conv1(outputs, outputs)
        outputs = self.up_conv2(outputs, u_connection_6)
        outputs = self.up_conv3(outputs, u_connection_5)
        outputs = self.up_conv4(outputs, u_connection_4)
        outputs = self.up_conv5(outputs, u_connection_3)
        outputs = self.up_conv6(outputs, u_connection_2)
        outputs = self.up_conv7(outputs, u_connection_1)

        outputs = self.final_conv_layer(outputs)
        return outputs
