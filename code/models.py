"""
models.py
Written by Adam Lavertu
Stanford University
"""

import torch
from torch import nn


class CheckNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CheckNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.dp1 = nn.Dropout(0.6)
        self.dp2 = nn.Dropout(0.4)
        self.fc = nn.Linear(1152, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        #         out = self.dp1(out)
        out = self.layer2(out)
        #         out = self.dp1(out)
        out = self.layer3(out)
        out = self.dp1(out)
        out = self.layer4(out)
        out = self.dp2(out)
        #         out = self.dp1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class CheckBoxEnsemble(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5):
        super(CheckBoxEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5

    def forward(self, x):
        x1 = self.model1(x.clone())
        x1 = x1.view(x1.size(0), -1)

        x2 = self.model2(x.clone())
        x2 = x2.view(x2.size(0), -1)

        x3 = self.model3(x.clone())
        x3 = x3.view(x3.size(0), -1)

        x4 = self.model4(x.clone())
        x4 = x4.view(x4.size(0), -1)

        x5 = self.model5(x.clone())
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return x


# Convolutional neural network (two convolutional layers)
class TemplateNet(nn.Module):
    def __init__(self, num_classes=1):
        super(TemplateNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #         self.fc = nn.Sequential(nn.Linear(2592, num_classes),
        #                        nn.ReLU())
        self.dp1 = nn.Dropout(0.2)
        self.fc = nn.Linear(113152, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dp1(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
