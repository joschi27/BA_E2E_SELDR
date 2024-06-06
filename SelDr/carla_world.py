import utils

utils.add_carla_to_path()

import carla


class CarlaWorld:
    def __init__(self, args):
        self.client = None
        self.world = None
        self.settings = None
        self.args = args

    def setup(self):
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        # Todo: add a check if the current world is called "TrainingMap". If not, load that.
        # sim_world = client.load_world('TrainingMap')

        # Set up world to be synchronous
        self.settings = self.world.get_settings()
        if not self.settings.synchronous_mode:
            self.settings.synchronous_mode = True
            self.settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(self.settings)

            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

    def get_carla_world(self):
        return self.world

    def tick(self):
        self.world.tick()

    def cleanup(self):
        # Make sure the simulation is in synchronous mode again, else unreal engine visuals freeze
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = None
        self.world.apply_settings(self.settings)

        # Destroy actors that were over from other failed attempts:
        actors = self.world.get_actors()
        vehicles = actors.filter('vehicle.*')

        for vehicle in vehicles:
            vehicle.destroy()