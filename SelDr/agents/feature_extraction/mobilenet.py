import carla
from torch import nn as nn
from torchvision import transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from hyperparameters import DEVICE


class MobileNetFeatureExtractionNetwork(nn.Module):

    image_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    def __init__(self) -> None:
        super(MobileNetFeatureExtractionNetwork, self).__init__()

        # Load the pre-trained MobileNet model
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features

        # Set the MobileNet to evaluation mode
        self.mobilenet.eval()

        # Disable gradient calculations
        for parameter in self.mobilenet.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # Extract features
        x = self.mobilenet(x)
        # Flatten the features to merge with state information later
        # Ensure the output is 1D by flattening, considering batch_size=1 for individual processing
        return x.view(-1)

    @staticmethod
    def to_tensor_input(image):
        # Apply the transformation
        input_tensor = MobileNetFeatureExtractionNetwork.image_transforms(image).to(DEVICE)

        # Add a batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor
