import torch
import torch.nn as nn
from ultralytics import YOLO

from hyperparameters import DEVICE


class YoloFeatures(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = YOLO("./Models/cone_detector.pt").to(DEVICE)
        self.yolo_features = torch.nn.Sequential(*list(self.model.model.children())[0][:-13]).to(DEVICE)

    def forward(self, x):
        with torch.no_grad():
            #pred = self.model.predict(source=x, device=DEVICE, show=True)
            features = self.yolo_features(x)
            return features.flatten()

    def to_tensor_input(self, x):
        return x
