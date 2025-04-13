# model.py
import torch.nn as nn
from torchvision import models
import torch

class CovidXRayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self._model.parameters():
            param.requires_grad = False

        self._model.features[28].requires_grad = True
        self._model.classifier[0].requires_grad = True
        self._model.classifier[3].requires_grad = True
        self._model.classifier[6] = nn.Linear(4096, 4) # 4 is the number of the classes
        
        
    def forward(self, image):
        return self._model(image)

class DualCovidXRayModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        for param in vgg16.parameters():
            param.requires_grad = False
            
        self.features = vgg16.features
        self.fc = vgg16.classifier
        for param in self.fc:
            param.requires_grad = True
            
        self.fc[0] = nn.Linear(512 * 7 * 7 * 2, 4096)
        self.fc[-1] = nn.Linear(self.fc[-1].in_features, num_classes)
        
    def forward(self, image1, image2):
        
        features1 = self.features(image1)
        features2 = self.features(image2)
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        
        combined_features = torch.cat((features1, features2), dim=1)
        output = self.fc(combined_features)
        return output