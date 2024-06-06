import json
import math
import random
from dataclasses import dataclass

import carla
import torch
import numpy as np
from carla import Image

import hyperparameters
from action_space import ActionSpace
from hyperparameters import DEVICE, CAMERA_FOV
from utils import convert_carla_image_to_PIL, replace_alpha_with_depth
from utils.CarDebugInfo import CarDebugInfo

from PIL import Image as PILImage

@dataclass
class AgentSensorData:
    cam_image: PILImage
    accelerometer_vector: any
    start_location: any
    current_location: any
    current_rotation: any
    has_collided: any
    velocity: any
    last_location: any
    relative_location: any
    last_velocity: any
    last_acceleration: any
    wheel_steering_angles: any
    cone_positions: any


class CarlaAgent:

    MAX_STEER_ANGLE = 50.0

    WORLD_CONES = None

    def __init__(self, world, args, agent_index=-1, camera_enabled=False, enable_debug_display=False):
        self.world = world
        self.vehicle = None
        self.camera = None
        self.depth_camera = None
        self.imu_sensor = None
        self.collision_sensor = None
        self.args = args
        self.spawn_point = None
        self.agent_index = agent_index
        self.enable_debug_display = enable_debug_display
        self.supress_depth = False

        self.relative_camera_location = carla.Transform(carla.Location(x=-0.1, z=1.00))

        self.debug_instance = CarDebugInfo()

        self.setup_vehicle()
        # if cameras in general are disabled but this one should still show only this agent needs a camera
        use_camera = camera_enabled
        if enable_debug_display:
            use_camera = True
        self.setup_sensors(use_camera)

        self.latest_camera_image = None
        self.latest_depth_image = None
        self.has_collided = False
        self.latest_accelerometer_data = None

        self.start_location = None
        self.last_location = None

        self.last_acceleration = [[0,0,0]]
        self.last_velocity = self.vehicle.get_velocity()

        self.debug_info = CarDebugInfo()

        self.supress_camera = False

    def setup_vehicle(self):
        # Code to spawn and set up the vehicle
        blueprint_library = self.world.get_blueprint_library()
        car_bp = blueprint_library.find('vehicle.zur.zur')
        spawn_points = self.world.get_map().get_spawn_points()

        if self.agent_index == -1:
            self.spawn_point = random.choice(spawn_points)
        else:
            self.spawn_point = spawn_points[self.agent_index]

        self.vehicle = self.world.spawn_actor(car_bp, self.spawn_point)
        self.vehicle.set_autopilot(False)

    def setup_sensors(self, with_camera):
        # Code to set up sensors like the camera, IMU, collision sensor, etc.

        blueprint_library = self.world.get_blueprint_library()

        if with_camera:
            # Add camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', f'{self.args.width}')
            camera_bp.set_attribute('image_size_y', f'{self.args.height}')
            camera_bp.set_attribute('fov', str(CAMERA_FOV))
            self.camera = self.world.spawn_actor(camera_bp, self.relative_camera_location, attach_to=self.vehicle,
                                                 attachment_type=carla.AttachmentType.Rigid)
            if not self.supress_depth:
                depth_bp = blueprint_library.find('sensor.camera.depth')
                #TODO: For speed, we could lower the resolution of the depth massively.
                depth_bp.set_attribute('image_size_x', f'{self.args.width}')
                depth_bp.set_attribute('image_size_y', f'{self.args.height}')
                depth_bp.set_attribute('fov', str(CAMERA_FOV))

                self.depth_camera = self.world.spawn_actor(depth_bp, self.relative_camera_location, attach_to=self.vehicle,
                                                     attachment_type=carla.AttachmentType.Rigid)

            self.start_camera()

        # Add collision sensor
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, collision_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        # Add accelerometer sensor
        imu_sensor_bp = blueprint_library.find('sensor.other.imu')

        # Example attributes, modify as needed
        # TODO: These attribute don't seem to exist - we need to get the zed camera accelerometer data properties and set them here in the future.
        # imu_sensor_bp.set_attribute('accelerometer_range', '100')  # Range in m/s^2 ->
        # imu_sensor_bp.set_attribute('accelerometer_noise_stddev_x', '0.0')  # Noise standard deviation
        # imu_sensor_bp.set_attribute('accelerometer_noise_stddev_y', '0.0')
        # imu_sensor_bp.set_attribute('accelerometer_noise_stddev_z', '0.0')
        self.imu_sensor = self.world.spawn_actor(imu_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.imu_sensor.listen(lambda data: self.process_imu_data(data))

    def start_camera(self):
        self.camera.listen(lambda image: self.camera_callback(image) if not self.supress_camera else None)

        if not self.supress_depth:
            self.depth_camera.listen(lambda image: self.depth_callback(image) if not self.supress_camera else None)

    def set_control_inputs(self, action: ActionSpace):
        control_input = action.convert_to_carla()
        self.vehicle.apply_control(control_input)


    # This method needs the sim world to have ticked before!
    def get_sensor_data(self) -> AgentSensorData:
        accelerometer_vector = np.array(
            [[self.latest_accelerometer_data.x, self.latest_accelerometer_data.y, self.latest_accelerometer_data.z]])
        last_acceleration = self.last_acceleration
        self.last_acceleration = accelerometer_vector

        velocity = self.vehicle.get_velocity()
        last_velocity = self.last_velocity
        self.last_velocity = velocity

        # Log the speed in the debug array so later, avg speed can be calculated.
        if self.debug_instance.get_debug_info("speed_array") is None:
            self.debug_instance.set_debug_info("speed_array", [])

        speed_array = self.debug_instance.get_debug_info("speed_array")
        speed_array.append(velocity.length())

        last_location = self.last_location
        self.last_location = self.vehicle.get_location()

        transform = self.vehicle.get_transform()

        relative_location = transform.location - self.start_location

        wheel_fl_angle = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        wheel_fr_angle = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        normalized_steer_angles = (wheel_fl_angle / CarlaAgent.MAX_STEER_ANGLE, wheel_fr_angle / CarlaAgent.MAX_STEER_ANGLE)

        pil_image = None
        if self.latest_camera_image is not None:
            self.debug_info.set_image_dimensions(self.args.width, self.args.height)
            self.debug_info.set_image(self.latest_camera_image)

            if self.latest_depth_image is None:
                pil_image = convert_carla_image_to_PIL(self.latest_camera_image)
            else:
                pil_image = replace_alpha_with_depth(self.latest_camera_image, self.latest_depth_image)

        cones = None

        return AgentSensorData(pil_image,
                               accelerometer_vector,
                               self.start_location,
                               transform.location,
                               transform.rotation,
                               self.has_collided,
                               velocity,
                               last_location,
                               relative_location,
                               last_velocity,
                               last_acceleration,
                               normalized_steer_angles,
                               cones)

    def destroy(self):
        if self.camera is not None:
            self.camera.destroy()
        if self.imu_sensor is not None:
            self.imu_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()

        print("Agent destruction done!")

    def reset(self):
        self.has_collided = False

        self.vehicle.set_transform(self.spawn_point)

        physics_control = self.vehicle.get_physics_control()
        physics_control.velocity = carla.Vector3D(0, 0, 0)
        physics_control.angular_velocity = carla.Vector3D(0, 0, 0)
        self.vehicle.apply_physics_control(physics_control)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        self.start_location = self.spawn_point.location
        self.last_location = self.spawn_point.location

        self.supress_camera = False

    def set_done(self):
        self.supress_camera = True

    def set_relative_location(self, location, rotation):

        new_x = self.start_location.x + location[0]
        new_y = self.start_location.y + location[1]
        new_z = self.start_location.z  # No change in the Z-axis (elevation)

        new_transform = carla.Transform(carla.Location(x=new_x, y=new_y, z=new_z),
                                        carla.Rotation(yaw=rotation - 90, pitch=0, roll=0))
        self.vehicle.set_transform(new_transform)

        self.last_location = new_transform.location

    def on_collision(self, event):
        # event contains information about the collision
        self.has_collided = True

    def process_imu_data(self, data):
        # Accelerometer data can be accessed with data.accelerometer
        self.latest_accelerometer_data = data.accelerometer

    # Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
    def camera_callback(self, data):
        self.latest_camera_image = data

    def depth_callback(self, data):
        self.latest_depth_image = data