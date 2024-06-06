import argparse
import logging
import math

import numpy as np
import pygame
from pygame.locals import K_ESCAPE, KEYDOWN

import utils

utils.add_carla_to_path()

import carla

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        #default='1280x720',
        default='640x480', #vga -> provides good middleground
        help='window resolution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    client = carla.Client(args.host, args.port)
    client.set_timeout(2000.0)

    sim_world = client.get_world()
    if args.sync:
        original_settings = sim_world.get_settings()
        settings = sim_world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

    blueprint_library = sim_world.get_blueprint_library()
    car_bp = blueprint_library.find('vehicle.micro.microlino')
    spawn_points = sim_world.get_map().get_spawn_points()

    #spawn_point = random.choice(spawn_points)  # or choose a specific point
    
    spawn_point = spawn_points[1]

    vehicle = sim_world.spawn_actor(car_bp, spawn_point)
    vehicle.set_autopilot(False)

    #Add camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{args.width}')
    camera_bp.set_attribute('image_size_y', f'{args.height}')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.2))
    camera = sim_world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)

    #Add collision sensor
    collision_sensor_bp = blueprint_library.find('sensor.other.collision')
    collision_transform = carla.Transform()
    collision_sensor = sim_world.spawn_actor(collision_sensor_bp, collision_transform, attach_to=vehicle)
    collision_sensor.listen(lambda event: on_collision(event))

    #Add accelerometer sensor
    imu_sensor_bp = blueprint_library.find('sensor.other.imu')
    
    # Example attributes, modify as needed
    # TODO: These attribute don't seem to exist - we need to get the zed camera accelerometer data properties and set them here in the future.
    #imu_sensor_bp.set_attribute('accelerometer_range', '100')  # Range in m/s^2 -> 
    #imu_sensor_bp.set_attribute('accelerometer_noise_stddev_x', '0.0')  # Noise standard deviation
    #imu_sensor_bp.set_attribute('accelerometer_noise_stddev_y', '0.0')
    #imu_sensor_bp.set_attribute('accelerometer_noise_stddev_z', '0.0')
    imu_sensor = sim_world.spawn_actor(imu_sensor_bp, carla.Transform(), attach_to=vehicle)
    imu_sensor.listen(lambda data: process_imu_data(data))

    renderObject = RenderObject(args.width, args.height)

    # Initialise the display
    pygame.init()
    pygame.display.set_caption("CARLA Camera View")
    clock = pygame.time.Clock()
    gameDisplay = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # Draw black to the display
    gameDisplay.fill((0,0,0))
    gameDisplay.blit(renderObject.surface, (0,0))
    pygame.display.flip()

    camera.listen(lambda image: pygame_callback(image, renderObject))

    tick = 0
    closed = False
    while not closed:
        try:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)

            gameDisplay.blit(renderObject.surface, (0,0))
            pygame.display.flip()

            tick += 0.1
            if(tick >= 50):
                control_vehicle(tick, vehicle)

            for event in pygame.event.get():
                # If the window is closed, break the while loop
                if event.type == pygame.QUIT:
                    closed = True
                    break
                
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        closed = True
                        break

        except Exception:
            logging.error(Exception)

    camera.destroy()
    imu_sensor.destroy()
    collision_sensor.destroy()
    vehicle.destroy()
    pygame.quit()
    return      

def on_collision(event):
    # event contains information about the collision
    other_actor = event.other_actor
    print(f"Collision with {other_actor.type_id}")

def process_imu_data(data):
    # Accelerometer data can be accessed with data.accelerometer
    print(f"Accelerometer: {data.accelerometer}")

def control_vehicle(tick, vehicle):
    control = carla.VehicleControl()
    control.throttle = 1
    control.brake = 0
    control.steer = math.sin(0.5 * tick)  # Swerving logic
    vehicle.apply_control(control)

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):
    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]
    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

if __name__ == '__main__':

    main()
