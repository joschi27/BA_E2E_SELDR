class SkidPadNormalizer:
    position_min = {'x': -35, 'y': -5}
    position_max = {'x': 35, 'y': 35}

    accel_min = -50  # m/s^2 -> about 5g's
    accel_max = 50  # m/s^2

    rotation_min = 0  # degrees
    rotation_max = 360

    vel_min = -15  # m/s -> about 50kmh max
    vel_max = 15

    @staticmethod
    def normalize_relative_position(x, y):
        x_normalized = (x - SkidPadNormalizer.position_min['x']) / (
                SkidPadNormalizer.position_max['x'] - SkidPadNormalizer.position_min['x']) * 2 - 1
        y_normalized = (y - SkidPadNormalizer.position_min['y']) / (
                SkidPadNormalizer.position_max['y'] - SkidPadNormalizer.position_min['y']) * 2 - 1
        return x_normalized, y_normalized

    @staticmethod
    def normalize_carla_accelerometer(accel_vector):
        x_normalized = (accel_vector['x'] - SkidPadNormalizer.accel_min) / (
                SkidPadNormalizer.accel_max - SkidPadNormalizer.accel_min) * 2 - 1
        y_normalized = (accel_vector['y'] - SkidPadNormalizer.accel_min) / (
                SkidPadNormalizer.accel_max - SkidPadNormalizer.accel_min) * 2 - 1
        z_normalized = (accel_vector['z'] - SkidPadNormalizer.accel_min) / (
                SkidPadNormalizer.accel_max - SkidPadNormalizer.accel_min) * 2 - 1
        return {'x': x_normalized, 'y': y_normalized, 'z': z_normalized}

    @staticmethod
    def normalize_carla_velocity(velocity_vector):
        x_normalized = (velocity_vector['x'] - SkidPadNormalizer.vel_min) / (
                SkidPadNormalizer.vel_max - SkidPadNormalizer.vel_min) * 2 - 1
        y_normalized = (velocity_vector['y'] - SkidPadNormalizer.vel_min) / (
                SkidPadNormalizer.vel_max - SkidPadNormalizer.vel_min) * 2 - 1
        z_normalized = (velocity_vector['z'] - SkidPadNormalizer.vel_min) / (
                SkidPadNormalizer.vel_max - SkidPadNormalizer.vel_min) * 2 - 1
        return {'x': x_normalized, 'y': y_normalized, 'z': z_normalized}

    @staticmethod
    def normalize_carla_rotation(rotation_degrees):
        normalized_rotation = (rotation_degrees - SkidPadNormalizer.rotation_min) / (
                SkidPadNormalizer.rotation_max - SkidPadNormalizer.rotation_min) * 2 - 1
        return normalized_rotation
