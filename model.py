import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
NUM_CLASSES =22
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-train model(whose weights has been downloaded)
resnet = models.resnet18().to(device)


    
class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.backbone = resnet

        self.classifier = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
    
class VGGNet(nn.Module):
    def __init__(self):	   #num_classes，此处为 二分类值为2
        super(VGGNet, self).__init__()
        net = models.vgg19_bn()   
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(    
                nn.Linear(512 * 7 * 7, 512), 
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, 22),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 
    
googlenet = models.googlenet().to(device)
class Googlenet(nn.Module):
    def __init__(self):
        super(Googlenet, self).__init__()
        self.backbone = googlenet

        self.classifier = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

