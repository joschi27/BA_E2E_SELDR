import argparse

import utils
from environments import MultiAgentEnvSkidPad, MultiAgentEnvAcceleration
from environments.multi_agent_env_trackdrive import MultiAgentEnvTrackDrive

utils.add_carla_to_path()

from train import TrainingRunner
from hyperparameters import NUM_AGENTS, MODE, SHOW_PYGAME_SCREEN
from mode import Mode


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
        # default='1280x720',
        default='512x512',
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
        '--continue-training',
        default=None,
        type=str,
        help='Model path to continue training on (default: None)')
    argparser.add_argument(
        '--no-logs',
        default=False,
        type=bool,
        help='Turn off logs (default=False)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    #env = MultiAgentEnvTrackDrive(args, NUM_AGENTS, MODE, SHOW_PYGAME_SCREEN, True)
    # env = MultiAgentEnvSkidPad(args, NUM_AGENTS, MODE, SHOW_PYGAME_SCREEN, True)
    env = MultiAgentEnvAcceleration(args, NUM_AGENTS, MODE, SHOW_PYGAME_SCREEN)

    runner = TrainingRunner(env, args.continue_training, args.no_logs)

    try:
        runner.run_training()
    finally:
        env.cleanup()
        print("Simulation ended")

if __name__ == '__main__':
    main()