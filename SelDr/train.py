import os
import time
from collections import deque
from datetime import datetime
from itertools import count

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import hyperparameters
from action_space import ActionSpace
from agents.PPO import PPO
from agents.feature_extraction import YOLOConeDetector, VGG16FeatureExtractionNetwork, MobileNetFeatureExtractionNetwork
from hyperparameters import *
from utils.CarDebugInfo import CarDebugInfo


class TrainingRunner(object):

    def __init__(self, environment, load_model_path: str, no_logs: bool = False):
        self.environment = environment
        self.load_model_path = load_model_path
        self.model_folder_path = datetime.now().strftime("%Y-%m-%d")
        self.create_logs = not no_logs

        self.debug_instance = CarDebugInfo()

    def run_training(self):

        past_features = [deque(maxlen=4) for _ in range(NUM_AGENTS)]  # Initialize a deque for each agent

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time = time.time()

        if self.create_logs:
            # setup tensorboard writer
            loaded = "_continued" if self.load_model_path else ""
            writer = SummaryWriter(
                log_dir=f"{TENSORBOARD_DIR}/run_{current_time}_{type(self.environment).__name__}{loaded}")

        feature_extractor = None

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
        ppo_agent = PPO(state_dim, action_dim, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, True, ACTION_STD,
                        NUM_AGENTS)

        if self.load_model_path:
            ppo_agent.load(self.load_model_path)

        total_steps_done = 0

        for i_episode in range(NUM_EPISODES):
            states = self.environment.reset()
            total_episode_rewards = [0] * NUM_AGENTS
            steps_survived = [0] * NUM_AGENTS

            print("Episode #", i_episode)

            agents_done = [False] * NUM_AGENTS
            update_queued = False
            for i_step in count():
                total_steps_done += 1

                # only set an action for the agents that are not yet done
                actions = [None] * NUM_AGENTS
                for i, (agent_done, state) in enumerate(zip(agents_done, states)):
                    if agent_done:
                        continue

                    # for vision, state is a tuple with image and sensor data
                    if MODE == Mode.VISION:
                        current_features = feature_extractor(feature_extractor.to_tensor_input(state[0]))

                        if i == hyperparameters.SPECTATOR_AGENT:
                            self.debug_instance.set_debug_info("extracted_features", current_features)

                        # Concatenate current features with past features
                        all_features = [current_features] + list(past_features[i])
                        # Ensure we always have exactly four past states (5 total states including the current)
                        if len(all_features) < 5:
                            # Pad with tensors of zeros if there are fewer than four past features
                            all_features += [torch.zeros_like(current_features) for _ in
                                             range(5 - len(all_features))]

                        # Now concatenate all features into one tensor
                        padded_features = torch.cat(all_features)

                        # Update past features deque
                        past_features[i].appendleft(current_features.clone().detach())

                        sensor_tensor = torch.tensor(state[1], dtype=torch.float, device=DEVICE)
                        merged = torch.cat((sensor_tensor, padded_features), 0)
                        actions[i] = ActionSpace(*ppo_agent.select_action(merged, i))
                    elif MODE == Mode.NUMERIC:
                        sensor_tensor = torch.tensor(state, dtype=torch.float, device=DEVICE)
                        actions[i] = ActionSpace(*ppo_agent.select_action(sensor_tensor, i))


                next_states, rewards, dones, truncateds = self.environment.step(actions)

                # add step rewards to agents' respective buffers
                for i, (reward, truncated, done) in enumerate(zip(rewards, truncateds, dones)):
                    if not agents_done[i]:
                        steps_survived[i] += 1  # Update survived steps for active agents
                    if agents_done[i]:
                        continue

                    ppo_agent.buffers[i].rewards.append(reward)
                    ppo_agent.buffers[i].is_terminals.append(truncated or done)

                    total_episode_rewards[i] += reward
                    agents_done[i] = truncated or done

                states = next_states

                # update PPO agent
                if total_steps_done % UPDATE_FREQUENCY == 0:
                    update_queued = True
                    print("Update queued")
                    total_time = time.time() - start_time
                    print(f"Steps per second: {total_steps_done / total_time}")

                # if continuous action space; then decay action std of output action distribution
                if total_steps_done % ACTION_STD_DECAY_FREQ == 0:
                    ppo_agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

                if all(agents_done):
                    break

            if update_queued:
                print("Update network")
                ppo_agent.update()

            if self.create_logs:
                writer.add_scalar("reward_avg", np.mean(total_episode_rewards), i_episode)
                writer.add_scalar("reward_std", np.std(total_episode_rewards), i_episode)
                writer.add_scalar("action_std", ppo_agent.action_std, i_episode)

                # Calculate the average survived steps
                avg_survived_steps = np.mean([step for step, done in zip(steps_survived, agents_done)])
                writer.add_scalar("survived_steps_avg", avg_survived_steps, i_episode)

                speed_array = self.debug_instance.get_debug_info("speed_array")
                writer.add_scalar("speed_avg_ms", np.mean(speed_array), i_episode)
                self.debug_instance.set_debug_info("speed_array", [])

                writer.flush()

            print("Total_Reward: ", total_episode_rewards, " - Episode: ", i_episode)

            if i_episode % SAVE_FREQUENCY == 0 and i_episode != 0:
                folder_path = f"./Models/{self.model_folder_path}_{current_time}"
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                ppo_agent.save(f"{folder_path}/ppo_agent_episode_{i_episode}_{type(self.environment).__name__}.h5")

        if self.create_logs:
            writer.close()
