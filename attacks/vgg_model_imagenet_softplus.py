'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16_softplus(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG16_softplus, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, num_classes),
        # )
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(4096, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.softplus(out, beta=10)
        out = self.conv2(out)
        out = F.softplus(out, beta=10)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        ##
        out = self.conv3(out)
        out = F.softplus(out, beta=10)
        out = self.conv4(out)
        out = F.softplus(out, beta=10)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        ##
        out = self.conv5(out)
        out = F.softplus(out, beta=10)
        out = self.conv6(out)
        out = F.softplus(out, beta=10)
        out = self.conv7(out)
        out = F.softplus(out, beta=10)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        ##
        out = self.conv8(out)
        out = F.softplus(out, beta=10)
        out = self.conv9(out)
        out = F.softplus(out, beta=10)
        out = self.conv10(out)
        out = F.softplus(out, beta=10)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        ##
        out = self.conv11(out)
        out = F.softplus(out, beta=10)
        out = self.conv12(out)
        out = F.softplus(out, beta=10)
        out = self.conv13(out)
        out = F.softplus(out, beta=10)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        ##
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        ##
        out = self.fc1(out)
        out = F.softplus(out, beta=10)
        out = self.drop1(out)
        out = self.fc2(out)
        out = F.softplus(out, beta=10)
        out = self.drop2(out)
        out = self.fc3(out)
        return out

    # def _make_layers(self, cfg):
    #     layers = []
    #     in_channels = 3
    #     for x in cfg:
    #         if x == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
    #                        # nn.BatchNorm2d(x),
    #                        # nn.ReLU(inplace=True)]
    #             in_channels = x
    #     # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    #     return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
