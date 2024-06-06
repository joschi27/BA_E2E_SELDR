import carla
import numpy as np
import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt
from torch.nn import functional as F
from ultralytics import YOLO

import hyperparameters
from hyperparameters import DEVICE


class YOLOConeDetector(torch.nn.Module):
    def __init__(self, model_path="./Models/cone_detector.pt") -> None:
        super(YOLOConeDetector, self).__init__()

        # Initialize and load the YOLO model
        self.model = YOLO(model_path)

    def forward(self, x):
        with torch.no_grad():
            rgb = x.convert("RGB")
            results = self.model.predict(source=rgb, device=DEVICE, verbose=False)

            # Assuming alpha channel is the last channel in your image x
            alpha = np.array(x.split()[-1])  # Convert PIL image's alpha channel to numpy array

            # Initialize an empty list for features; this will hold the info
            features_list = []

            if len(results[0].boxes.xyxy) > 0:
                for box, cls in zip(results[0].boxes.xyxyn[:, :4], results[0].boxes.cls):
                    # Calculate center of the box
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2

                    # Query the alpha channel for distance at the center of the box
                    distance = alpha[int(center_y * alpha.shape[0]), int(center_x * alpha.shape[1])]

                    # Normalize the distance to 0-1
                    normalized_distance = (distance / 255)

                    # Append normalized center_x, center_y, class, and distance to features_list
                    features_list.append([center_x.item(), center_y.item(), cls.item(), normalized_distance])

                # Convert features_list to tensor
                features = torch.tensor(features_list, dtype=torch.float32, device=DEVICE)

                # Pad the features tensor to ensure it has space for n detections
                if features.size(0) < hyperparameters.NUM_DETECTIONS:
                    padding = hyperparameters.NUM_DETECTIONS - features.size(0)
                    features = F.pad(features, (0, 0, 0, padding), "constant", 0)

            else:
                # Return a zero tensor for 10 detections * (2 coords + class + distance) = 40 elements
                features = torch.zeros((hyperparameters.NUM_DETECTIONS, 4), device=DEVICE)

            return features.flatten()[:hyperparameters.NUM_DETECTIONS * 4]  # Ensure tensor is flattened and has a fixed size

    @staticmethod
    def to_tensor_input(image):
        return image

    def debug_draw_points(self, image, features, class_names):
        rgb_img = image.convert("RGB")
        draw = ImageDraw.Draw(rgb_img)
        for i in range(0, len(features), 4):
            if features[i + 3] > 0:  # If distance is not zero
                # Draw point at center_x, center_y
                x, y = features[i] * rgb_img.width, features[i + 1] * rgb_img.height
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')

                # Draw class and distance
                class_id = int(features[i + 2])
                distance = features[i + 3]
                draw.text((x, y), f"{class_names[class_id]}, {distance}", fill='red')

        plt.imshow(rgb_img)
        plt.axis('off')
        plt.show()
