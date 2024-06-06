from matplotlib import pyplot as plt

import hyperparameters
import utils

utils.add_carla_to_path()

import carla
import numpy as np
from PIL import Image
import torch

def convert_carla_image_to_PIL(image: carla.Image):
    image_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))

    # Convert BGRA to RGB. np.ndarray is laid out in row-major order, so the last axis is the one we want to reverse.
    # We discard the alpha channel with '...,:-1' which is the numpy way to apply to all preceding axes.
    image_rgb = image_bgra[:, :, :3][:, :, ::-1]

    pil_image = Image.fromarray(image_rgb)
    return pil_image


def decode_depth_image(carla_depth_image, min_depth=0.3):
    """
    Decodes a CARLA depth image to normalized depth values in the range [0, 1],
    where 0 corresponds to min_depth meters and 1 corresponds to max_depth meters.
    """
    max_depth = hyperparameters.MAX_CONE_DISTANCE

    depth_data = np.frombuffer(carla_depth_image.raw_data, dtype=np.uint8)
    depth_data = depth_data.reshape((carla_depth_image.height, carla_depth_image.width, 4))

    # Cropping
    top_crop = int(carla_depth_image.height * 0.50)  # Top 50%
    bottom_crop = int(carla_depth_image.height * 0.80)  # Keep until 80% from the top, removing bottom 20%
    depth_data = depth_data[top_crop:bottom_crop, :, :]

    # Decode depth from the RGB channels.
    depth_in_meters = depth_data[:, :, :3].astype(np.float32)
    depth_in_meters = depth_in_meters[:, :, 2] + depth_in_meters[:, :, 1] * 256 + depth_in_meters[:, :, 0] * 256**2
    depth_in_meters /= (256**3 - 1)
    depth_in_meters *= 1000  # Scale to meters if necessary.

    # Normalize depth values to the range [0, 1] based on specified min and max depths.
    normalized_depth = (depth_in_meters - min_depth) / (max_depth - min_depth)
    normalized_depth = np.clip(normalized_depth, 0, 1)  # Ensure values fall within [0, 1].

    return normalized_depth


def replace_alpha_with_depth(carla_rgb_image, carla_depth_image):
    """
    Replaces the alpha channel of an RGBA image from CARLA with the depth information from a CARLA depth sensor image.
    """
    # Convert RGB image data from BGRA to RGBA.
    rgb_data = np.frombuffer(carla_rgb_image.raw_data, dtype=np.uint8)
    rgb_data = rgb_data.reshape((carla_rgb_image.height, carla_rgb_image.width, 4))
    rgba_image = rgb_data[:, :, [2, 1, 0, 3]]  # Reordering BGRA to RGBA

    # Cropping
    top_crop = int(carla_rgb_image.height * 0.50)  # Top 50%
    bottom_crop = int(carla_rgb_image.height * 0.80)  # Keep until 80% from the top, removing bottom 20%
    rgba_image = rgba_image[top_crop:bottom_crop, :, :]

    # Decode the depth image to get depth in meters.
    normalized_depth = decode_depth_image(carla_depth_image)

    # Normalize the depth data to 0-255 and replace the alpha channel.
    rgba_image[:, :, 3] = (normalized_depth * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgba_image, 'RGBA')

    return pil_img


    #TODO: CREATE IMAGE USING THIS AND UPLOAD TO BA!
    def visualize_feature_maps(self, x, layer_idx=0, n_images=5):
        """
        Visualizes the feature maps of a given layer.
        :param x: The input tensor after transformations, expected shape (1, C, H, W)
        :param layer_idx: Index of the layer to visualize the features from.
        :param n_images: Number of feature maps to visualize.
        """
        with torch.no_grad():
            # Assuming x is already a tensor from to_tensor_input method
            # Forward pass through the mobilenet up to the specified layer
            for i, layer in enumerate(self.mobilenet.children()):
                x = layer(x)
                if i == layer_idx:
                    break

            # Take the first n_images feature maps
            x = x[0, :n_images, :, :].cpu()

            fig, axs = plt.subplots(1, n_images, figsize=(20, 10))
            for i, ax in enumerate(axs.flat):
                ax.imshow(x[i], cmap='viridis')
                ax.axis('off')
            plt.show()
