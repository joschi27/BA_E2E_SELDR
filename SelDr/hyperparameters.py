import torch

from mode import Mode

############################### General Settings ###############################
NUM_EPISODES = 10000
NUM_AGENTS = 20
MAX_EPISODE_LENGTH = 250
UPDATE_FREQUENCY = MAX_EPISODE_LENGTH * 3

################################# PPO Settings #################################
GAMMA = 0.90                        # reward discount factor
LR_ACTOR = 0.0005
LR_CRITIC = 0.0005
K_EPOCHS = 10
EPS_CLIP = 0.2
ACTION_STD = 0.6                    # starting std for action distribution (Multivariate Normal)
ACTION_STD_DECAY_RATE = 0.0002       # linearly decay action_std (action_std = action_std - action_std_decay_rate)
MIN_ACTION_STD = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
ACTION_STD_DECAY_FREQ = 1000

#################################### config ####################################
TENSORBOARD_DIR = "./logs"
SAVE_FREQUENCY = 100
SHOW_PYGAME_SCREEN = True
SPECTATOR_AGENT = 0                 # index of the agent you want to watch, SHOW_PYGAME_SCREEN must be True
CAMERA_FOV = 90
MAX_CONE_DISTANCE = 20                    # Maximum distance a cone can be from the car to be picked up by depth
NUM_DETECTIONS = 10                       # The amount of cone detections from the YOLO network

MODE = Mode.VISION                 # Switch between vision and numeric state mode

################################## set device ##################################
# set device to cpu or cuda
DEVICE = torch.device('cpu')
if (torch.cuda.is_available()):
    DEVICE = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(DEVICE)))
else:
    print("Device set to : cpu")
