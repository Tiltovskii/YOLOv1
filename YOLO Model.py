import torch
from torch import nn


class CNNBlock(nn.Module):  # можно поменять на Lightning
    def __init__(self, in_channels, out_channels, is_max_pool: bool = False, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)  # в статье еще не знали про батчнорм, но мы то из будущего ...
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.is_maxpool = is_max_pool  # не после каждой свертки нужно делать maxpool
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.leakyrelu(self.batchnorm(self.conv(x)))

        if self.is_maxpool:
            x = self.maxpool(x)

        return x


class YOLO(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        """
        :param: S * S - количество ячеек на которые разбивается изображение
        :param: B - количество предсказанных прямоугольников в каждой ячейке
        :param: C - количество классов
        """

        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        self.conv1 = nn.Sequential(
            CNNBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, is_max_pool=True),
            # [64, 112, 112]
            CNNBlock(in_channels=64, out_channels=192, kernel_size=3, padding=1, is_max_pool=True)  # [192, 56, 56]
        )

        self.conv2 = nn.Sequential(
            CNNBlock(in_channels=192, out_channels=128, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=256, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, is_max_pool=True),  # [512, 28, 28]
        )

        self.conv3 = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=512, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, is_max_pool=True)  # [1024, 14, 14]
        )

        self.conv4 = nn.Sequential(
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, is_max_pool=False),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, is_max_pool=False)
            # [1024, 7, 7]
        )

        self.conv5 = nn.Sequential(
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, is_max_pool=False),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, is_max_pool=False)  # [1024, 7, 7]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2, inplace=False),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.fc(x)

        x = x.reshape(-1, self.S, self.S, 5 * self.B + self.C)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    grid_size = 7
    examples_per_cell = 2
    nums_of_classes = 3

    temp_model = YOLO(S=grid_size, B=examples_per_cell, C=nums_of_classes)
    expected_output_shape = temp_model.S * temp_model.S * (5 * temp_model.B + temp_model.C)
    temp_data = torch.zeros(1, 3, 448, 448)

    assert temp_model(temp_data).reshape(-1).shape[0] == expected_output_shape
