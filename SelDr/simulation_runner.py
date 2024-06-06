import json
import os
from collections import deque

import torch

import hyperparameters
from agents.ActorCritic import ActorCritic
from action_space import ActionSpace

import argparse

import utils
from agents.feature_extraction import YOLOConeDetector
from environments import MultiAgentEnvTrackDrive, MultiAgentEnvAcceleration, MultiAgentEnvSkidPad
from mode import Mode
import torch.nn.functional as F

from utils.CarDebugInfo import CarDebugInfo

utils.add_carla_to_path()


from hyperparameters import MODE, DEVICE


class SimulationRunner:
    def __init__(self, checkpoint_path, environment, has_continuous_action_space=True):
        self.has_continuous_action_space = has_continuous_action_space
        self.environment = environment

        self.metrics = {'completion_rate': [], 'average_speed': [], 'speed_fluctuations': [], 'crash_rate': [], 'crash_locations': [], 'path_with_speeds': []}
        self.log_dir = 'metrics'
        self.checkpoint_path = checkpoint_path

        self.debug_instance = CarDebugInfo()

        # for vision
        if MODE == Mode.VISION:
            feature_extractor = YOLOConeDetector().to(DEVICE)
            initial_image, initial_state = self.environment.reset()[0]

            # perform one forward pass through the feature extraction and merge to get dimensions
            features = feature_extractor(feature_extractor.to_tensor_input(initial_image))

            desired_length_to_add = features.numel() * 4
            padded_features = F.pad(features, (0, desired_length_to_add), "constant", 0)

            state = torch.tensor(initial_state, dtype=torch.float, device=DEVICE)
            merged = torch.cat((state, padded_features), 0)
            state_dim = merged.shape[0]
        elif MODE == Mode.NUMERIC:
            state_dim = torch.tensor(self.environment.reset()[0]).shape[0]
        else:
            raise ValueError(f"Current mode ({MODE}) not supported")

        action_dim = ActionSpace.get_output_dimension()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, 0.001).to(DEVICE)

        # Load the trained weights
        self.load(checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def select_action(self, state):
        """
        Selects an action based on the current state using the trained policy.
        """
        with torch.no_grad():
            action, _, _ = self.policy.act(state)

            if self.has_continuous_action_space:
                return action.detach().cpu().numpy().flatten()
            else:
                return action.item()

    def run_simulation(self, create_metrics=False):

        past_features = deque(maxlen=4)  # Initialize a deque for each agent

        for i in range(100):

            states = self.environment.reset()
            state = states[0]
            agents_done = [False] * 1
            total_reward = 0

            #metrics
            metric_path_with_speeds = []

            feature_extractor = YOLOConeDetector().to(DEVICE)

            while not agents_done[0]:
                action = None
                # for vision, state is a tuple with image and sensor data
                if MODE == Mode.VISION:
                    current_features = feature_extractor(feature_extractor.to_tensor_input(state[0]))

                    self.debug_instance.set_debug_info("extracted_features", current_features)

                    # Concatenate current features with past features
                    all_features = [current_features] + list(past_features)
                    # Ensure we always have exactly four past states (5 total states including the current)
                    if len(all_features) < 5:
                        # Pad with tensors of zeros if there are fewer than four past features
                        all_features += [torch.zeros_like(current_features) for _ in
                                         range(5 - len(all_features))]

                    # Now concatenate all features into one tensor
                    padded_features = torch.cat(all_features)

                    # Update past features deque
                    past_features.appendleft(current_features.clone().detach())

                    sensor_tensor = torch.tensor(state[1], dtype=torch.float, device=DEVICE)
                    merged = torch.cat((sensor_tensor, padded_features), 0)
                    action = self.select_action(merged)
                elif MODE == Mode.NUMERIC:
                    sensor_tensor = torch.tensor(state, dtype=torch.float, device=DEVICE)
                    action = self.select_action(sensor_tensor)

                states, rewards, dones, truncateds = self.environment.step([ActionSpace(*action)])
                total_reward += rewards[0]
                agents_done[0] = dones[0] or truncateds[0]
                state = states[0]

                if create_metrics:
                    if self.debug_instance.get_debug_info("metric_crashed"):
                        self.metrics["crash_locations"].append((
                            self.debug_instance.get_debug_info("metric_current_rel_location").x,
                            self.debug_instance.get_debug_info("metric_current_rel_location").y
                        ))

                    metric_path_with_speeds.append((
                        self.debug_instance.get_debug_info("metric_current_rel_location").x,
                        self.debug_instance.get_debug_info("metric_current_rel_location").y,
                        self.debug_instance.get_debug_info("current_speed")
                    ))

            if create_metrics:
                self.metrics['completion_rate'].append(self.debug_instance.get_debug_info("metric_current_track_completion"))
                self.metrics['path_with_speeds'].append(metric_path_with_speeds)

                self.metrics['crash_rate'].append(self.debug_instance.get_debug_info("metric_crashed"))

                speeds = self.debug_instance.get_debug_info("speed_array")
                self.metrics['average_speed'].append(sum(speeds) / len(speeds))

            print(f"Total reward from simulation: {total_reward}")

    def save_metrics(self):
        # Construct the list of metrics for each run
        metrics_data = []
        for i in range(len(self.metrics['completion_rate'])):
            run_data = {
                'Run': i + 1,
                'Completion Rate': self.metrics['completion_rate'][i],
                'Average Speed': self.metrics['average_speed'][i],
                'Crashed': bool(self.metrics['crash_rate'][i])
            }

            # Add paths with speeds if available
            if i < len(self.metrics['path_with_speeds']):
                run_data['Path with Speeds'] = self.metrics['path_with_speeds'][i]

            # Add crash locations if available
            if i < len(self.metrics['crash_locations']):
                run_data['Crash Locations'] = self.metrics['crash_locations'][i]

            metrics_data.append(run_data)

        # Prepare the data dictionary
        data = {'metrics': metrics_data}

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Write the data to a JSON file
        json_file_path = os.path.join(self.log_dir, f'{os.path.basename(self.checkpoint_path)}_metrics.json')
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Metrics successfully saved to {json_file_path}")

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
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    create_eval_metrics = True

    env = MultiAgentEnvSkidPad(args, 1, MODE, True, False, enable_metric_logs=create_eval_metrics)
    #env = MultiAgentEnvTrackDrive(args, 1, MODE, True, random_start=False, enable_metric_logs=create_eval_metrics)
    #env = MultiAgentEnvAcceleration(args, 1, MODE, True, enable_metric_logs=create_eval_metrics)

    #checkpoint_path = "Models/2024-05-26_2024-05-26_16-42-09/ppo_agent_episode_900_MultiAgentEnvAcceleration.h5"
    #checkpoint_path = "Models/2024-05-20_2024-05-20_16-15-59/ppo_agent_episode_6000_MultiAgentEnvTrackDrive.h5"
    checkpoint_path = "Models/2024-05-26_2024-05-26_17-12-31/ppo_agent_episode_9900_MultiAgentEnvSkidPad.h5"

    try:
        runner = SimulationRunner(checkpoint_path, env)
        runner.run_simulation(create_eval_metrics)

        if create_eval_metrics:
            runner.save_metrics()

    finally:

        env.cleanup()
        print("Simulation ended")

if __name__ == '__main__':
    main()
