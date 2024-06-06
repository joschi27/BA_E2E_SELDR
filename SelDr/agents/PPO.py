import torch
import torch.nn as nn
import os

from agents.ActorCritic import ActorCritic
from hyperparameters import DEVICE


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, num_buffers=1):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffers = [RolloutBuffer() for i in range(num_buffers)]

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, idx_buffer = 0):
        # select buffer to write actions to
        buffer = self.buffers[idx_buffer]

        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(action_logprob)
            buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(action_logprob)
            buffer.state_values.append(state_val)

            return action.item()

    def update(self):

        # merge experienced trajectories from multiple buffers into single buffer
        buffer = RolloutBuffer()
        for agent_buffer in self.buffers:
            buffer.actions.extend(agent_buffer.actions)
            buffer.states.extend(agent_buffer.states)
            buffer.logprobs.extend(agent_buffer.logprobs)
            buffer.rewards.extend(agent_buffer.rewards)
            buffer.state_values.extend(agent_buffer.state_values)
            buffer.is_terminals.extend(agent_buffer.is_terminals)

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(DEVICE)
        old_state_values = torch.squeeze(torch.stack(buffer.state_values, dim=0)).detach().to(DEVICE)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffers
        for buffer in self.buffers:
            buffer.clear()

    def save(self, checkpoint_path):
        # Extract the directory path from the full checkpoint path
        directory_path = os.path.dirname(checkpoint_path)

        # Check if the directory already exists
        if not os.path.exists(directory_path):
            # Create the directory, including any intermediate directories
            os.makedirs(directory_path)
            print(f"Created directories for path: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")

        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
