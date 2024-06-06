import random

import utils

utils.add_carla_to_path()

import carla


class ActionSpace:
    def __init__(self, drive: float, steering: float):
        # A vector that goes from -1 to 1. -1 = breaking and +1 = full throttle. 0 = nothing
        self.drive = drive
        # Steering. -1 = full left, +1 = right
        self.steering = steering

    def set_action(self, drive: float, steering: float):
        self.drive = drive
        self.steering = steering

    # Normalizes the action output from the model ([-1 to 1] to three distinct values.)
    def normalize_action(self):
        steer = float(self.steering)  # Steering remains the same, as it's already in the range [-1, 1]
        throttle = max(0, float(self.drive))  # Positive number is throttle
        brake = -min(0, float(self.drive))  # Brake is the negative part of the number

        return steer, throttle, brake

    def convert_to_carla(self):
        """
        Process the action predicted by the neural network.
        action: A list or array of [steering, throttle, brake]
        Returns a carla.VehicleControl object.
        """
        steer, throttle, brake = self.normalize_action()

        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    @classmethod
    def from_tensor(cls, tensor):
        drive = tensor[0].item()
        steer = tensor[1].item()
        return cls(drive, steer)

    @staticmethod
    def get_output_dimension():
        # Returns the dimension of the output tensor expected by the neural network
        return 2

    @classmethod
    def random_sample(cls):
        steer = random.uniform(-1, 1)
        drive = random.uniform(-1, 1)
        return cls(steer, drive)
