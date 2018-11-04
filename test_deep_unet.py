import unittest

import torch
import deep_unet


class MyTestCase(unittest.TestCase):

    def test_donw_block_first_layer(self):
        down_block_first = deep_unet.DownBlock_FirstLayer(in_channels=3, out_channels=32, hidden_layer=[64, 64])
        sample_inputs = torch.rand((1, 3, 640, 640))
        output, u_connection = down_block_first(sample_inputs)
        self.assertTupleEqual(tuple(output.size()), (1, 32, 320, 320))
        self.assertTupleEqual(u_connection.size(), (1, 32, 640, 640))

    def test_down_block(self):
        down_block = deep_unet.DownBlock(in_channels=32, out_channels=32, hidden_layer=64)

        for size in [640, 320, 160, 80, 40, 20]:
            sample_inputs = torch.rand((1, 32, size, size))
            outputs, u_connection = down_block(sample_inputs)
            self.assertTupleEqual(outputs.size(), (1, 32, size//2, size//2))
            self.assertTupleEqual(u_connection.size(), (1, 32, size, size))

    def test_up_block(self):
        up_block = deep_unet.UpBlock(in_channels=32*2, out_channels=32, hidden_layer=64)
        for size in [10, 20, 40, 80, 160, 320]:
            sample_inputs1 = torch.rand((1, 32, size, size))
            sample_inputs2 = torch.rand((1, 32, size, size))
            outputs = up_block(sample_inputs1, sample_inputs2)
            self.assertTupleEqual(outputs.size(), (1, 32, size * 2, size * 2))

    def test_deep_unet(self):
        n_classes = 4
        model = deep_unet.DeepUnet(n_classes=n_classes)
        sample_inputs = torch.rand((1, 3, 640, 640))
        outputs = model(sample_inputs)
        self.assertTupleEqual(outputs.size(), (1, n_classes, 640, 640))


if __name__ == '__main__':
    unittest.main()
