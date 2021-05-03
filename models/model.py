import torch.nn as nn
import torch

class Resnet18(nn.Module):
    def __init__(self, model):
        super(Resnet18, self).__init__()
        # 拿掉model的最後一層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, 2) #加上一層參數修改好的全連接層
    
    def forward(self, x):
        x = self.resnet_layer(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc (x)

        return x

class MyResnet18(nn.Module):
    def __init__(self, model):
        super(MyResnet18, self).__init__()
        # 拿掉model的最後三層(layer4、average pooling、fully connected)
        self.resnet_layer = nn.Sequential(*list(model.children())[:-3])
        # ==================================================================================== #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2),
            nn.BatchNorm2d(512)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        # ==================================================================================== #
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet_layer(x)

        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        out = nn.ReLU()(out)
        identity = out
        out = self.conv3(out)
        out = self.conv4(out)
        out += identity
        out = nn.ReLU()(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc (out)

        return out
