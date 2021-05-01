import torch.nn as nn

class MyResnet18(nn.Module):
    def __init__(self, model):
        super(MyResnet18, self).__init__()
        # 取掉model的後1層
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, 2) #加上一層參數修改好的全連接層

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x