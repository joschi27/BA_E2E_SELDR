import math
from typing import List

import hyperparameters
from carla_world import CarlaWorld
from carla_agent import CarlaAgent, AgentSensorData
from action_space import ActionSpace
from utils.SkidPadDirection import SkidPadDirection
from hyperparameters import MAX_EPISODE_LENGTH
from mode import Mode
from reward.reward_function_skidpad import SkidPadRaceRewardFunction
from environments.SkidPadNormalizer import SkidPadNormalizer
from utils.CarDebugInfo import CarDebugInfo


class MultiAgentEnvSkidPad:

    def __init__(self, args, num_agents, mode: Mode, enable_debug_display, random_start=True, enable_metric_logs=False):
        self.num_agents = num_agents
        self.world = None
        self.carla_agents = None
        self.args = args
        self.mode = mode
        self.enable_debug_display = enable_debug_display
        self.reward_functions = None
        self.random_start = random_start
        self.current_episode_steps = 0

        self.last_states = None

        self.debug_instance = CarDebugInfo()

        self.evaluation_metrics_run = enable_metric_logs

    def reset(self):
        """
        Resets all agents and returns their states in a list
        """

        if not self.world:
            self.world = CarlaWorld(self.args)
            self.world.setup()

            carla_world = self.world.get_carla_world()
            self.carla_agents = [CarlaAgent(carla_world,
                                            self.args,
                                            agent_index,
                                            self.mode == Mode.VISION or self.mode == Mode.VISION_DEPTH,
                                            self.enable_debug_display and agent_index == hyperparameters.SPECTATOR_AGENT,
                                            )
                                 for agent_index in range(self.num_agents)]

            self.reward_functions = [SkidPadRaceRewardFunction() for _ in
                                     range(self.num_agents)]

        for carla_agent, reward_function in zip(self.carla_agents, self.reward_functions):
            carla_agent.reset()

            if self.random_start:
                point, rotation = reward_function.get_set_random_location()
                if point is not None:
                    carla_agent.set_relative_location(point, rotation)

        self.current_episode_steps = 0

        # Run a tick or maybe multiple so that agents are initialized and sitting idle
        for i in range(10):
            self.world.tick()

        states = []
        self.last_states = []
        for carla_agent in self.carla_agents:
            sensor_data = carla_agent.get_sensor_data()
            next_waypoint = (0.0, 0.0)

            state = self.create_state_for_current_mode(sensor_data, next_waypoint)
            # Append the steering state for skidpad
            state[1].append(int(SkidPadDirection.FORWARD.value))
            # Pad the state with 5 times the amount of values it has already, so we can later replace those values with
            # old states
            state_length = len(state[1]) * 4
            state[1].extend([0] * state_length)

            states.append(state)

        self.last_states = states

        return states

    def step(self, actions: List[ActionSpace]):
        """
        Performs a single step for each agent
        :param actions: List of actions, must match number of agents"""

        if len(actions) != self.num_agents:
            raise ValueError(f"Number of actions ({len(actions)}) does not match number of agents ({self.num_agents})")

        for action, carla_agent in zip(actions, self.carla_agents):
            if action is not None:
                carla_agent.set_control_inputs(action)

        self.world.tick()

        self.current_episode_steps += 1

        states, rewards, dones, truncateds = [None] * self.num_agents, [None] * self.num_agents, [
            None] * self.num_agents, [None] * self.num_agents,
        for i, (action, carla_agent, reward_function) in enumerate(
                zip(actions, self.carla_agents, self.reward_functions)):
            # skip if no action was provided
            if action is None:
                continue

            sensor_data = carla_agent.get_sensor_data()

            # Vector to next waypoint
            next_waypoint = reward_function.get_relative_vector_to_next_waypoint()
            state = self.create_state_for_current_mode(sensor_data, next_waypoint)

            #TODO: In the future, remove the steering direction from this state as this gets replicated n times.
            #Add steering direction to state
            state[1].append(reward_function.get_drive_direction_to_next_waypoint())

            # get the state of the last step except camera
            saved_sensor_state = self.last_states[i][1]

            # Append the last sensor step except camera
            current_sensor_state = state[1]

            combined_sensor_state = current_sensor_state + saved_sensor_state[:-len(current_sensor_state)]

            state = (state[0], combined_sensor_state)

            reward, done = reward_function.compute_reward(sensor_data.start_location,
                                                          sensor_data.last_location,
                                                          sensor_data.current_location,
                                                          sensor_data.has_collided,
                                                          sensor_data.current_rotation,
                                                          action,
                                                          sensor_data.velocity)

            if self.evaluation_metrics_run:
                self.debug_instance.set_debug_info("metric_current_track_completion", reward_function.get_track_completion_percentage())
                self.debug_instance.set_debug_info("metric_current_rel_location", sensor_data.relative_location)
                self.debug_instance.set_debug_info("metric_crashed", sensor_data.has_collided)
                self.debug_instance.set_debug_info("current_speed", sensor_data.velocity.length())

            if carla_agent.enable_debug_display:
                self.debug_instance.set_debug_info("last_reward", reward)
                self.debug_instance.set_debug_info("waypoint_vector",
                                                   reward_function.get_relative_vector_to_next_waypoint())
                self.debug_instance.set_debug_info("current_waypoint", reward_function.get_current_point())
                self.debug_instance.set_debug_info("relative_location_vector", sensor_data.relative_location)
                self.debug_instance.set_debug_info("predicted_action", action)
                self.debug_instance.set_debug_info("accelerometer_vector", sensor_data.accelerometer_vector)
                self.debug_instance.set_debug_info("wheel_steering_angles", sensor_data.wheel_steering_angles)
                self.debug_instance.set_debug_info("splines", reward_function.get_splines())
                self.debug_instance.set_debug_info("segment_splines", reward_function.get_segmented_splines())
                self.debug_instance.set_debug_info("current_rotation", sensor_data.current_rotation)
                self.debug_instance.set_debug_info("current_location", sensor_data.current_location)
                self.debug_instance.set_debug_info("driving_direction", reward_function.get_drive_direction_to_next_waypoint())

                self.debug_instance.render_debug_info()

            if sensor_data.has_collided or done:
                carla_agent.reset()
                carla_agent.set_done()
                reward_function.reset()

            states[i] = state
            rewards[i] = reward
            if self.evaluation_metrics_run:
                truncateds[i] = sensor_data.has_collided
            else:
                truncateds[i] = sensor_data.has_collided or self.current_episode_steps >= MAX_EPISODE_LENGTH
            dones[i] = done

        self.last_states = states

        return states, rewards, dones, truncateds

    def cleanup(self):
        if self.carla_agents is not None:
            for carla_agent in self.carla_agents:
                carla_agent.destroy()
        if self.world is not None:
            self.world.cleanup()

    def create_state_for_current_mode(self, sensor_data: AgentSensorData, next_waypoint=None):
        if self.mode == Mode.NUMERIC:
            state = self.create_numeric_state(sensor_data, next_waypoint)
        elif self.mode == Mode.VISION:
            state = self.create_vision_state(sensor_data)
        else:
            raise Exception("Unknown mode: {}".format(self.mode))

        return state

    @staticmethod
    def create_vision_state(sensor_data: AgentSensorData):
        """
        Vision state is a tuple of the latest camera image (for feature extraction)
        and an array of other sensor data
        """

        normalized_accel = SkidPadNormalizer.normalize_carla_accelerometer({'x': sensor_data.accelerometer_vector[0][0],
                                                                            'y': sensor_data.accelerometer_vector[0][1],
                                                                            'z': sensor_data.accelerometer_vector[0][
                                                                                2]})

        speed = math.sqrt(
            sensor_data.velocity.x ** 2 + sensor_data.velocity.y ** 2 + sensor_data.velocity.z ** 2) / 30  # normalize with max speed of about 100kmh

        return sensor_data.cam_image, [normalized_accel['x'], normalized_accel['y'], normalized_accel['z'],
                                       speed, sensor_data.wheel_steering_angles[0],
                                       sensor_data.wheel_steering_angles[1]]


    @staticmethod
    def create_numeric_state(sensor_data: AgentSensorData, next_waypoint):

        normalized_x, normalized_y = SkidPadNormalizer.normalize_relative_position(sensor_data.relative_location.x,
                                                                                   sensor_data.relative_location.y)

        normalized_pitch = SkidPadNormalizer.normalize_carla_rotation(sensor_data.current_rotation.pitch)
        normalized_yaw = SkidPadNormalizer.normalize_carla_rotation(sensor_data.current_rotation.yaw)
        normalized_roll = SkidPadNormalizer.normalize_carla_rotation(sensor_data.current_rotation.roll)

        normalized_accel = SkidPadNormalizer.normalize_carla_accelerometer({'x': sensor_data.accelerometer_vector[0][0],
                                                                            'y': sensor_data.accelerometer_vector[0][1],
                                                                            'z': sensor_data.accelerometer_vector[0][
                                                                                2]})

        normalized_velocity = SkidPadNormalizer.normalize_carla_velocity({'x': sensor_data.velocity.x,
                                                                          'y': sensor_data.velocity.y,
                                                                          'z': sensor_data.velocity.z})

        normalized_waypoint_x, normalized_waypoint_y = SkidPadNormalizer.normalize_relative_position(next_waypoint[0],
                                                                                                     next_waypoint[1])
        return [normalized_x,
                normalized_y,
                normalized_pitch,
                normalized_yaw,
                normalized_roll,
                normalized_accel['x'],
                normalized_accel['y'],
                normalized_accel['z'],
                normalized_velocity['x'],
                normalized_velocity['y'],
                normalized_velocity['z'],
                normalized_waypoint_x,
                normalized_waypoint_y]