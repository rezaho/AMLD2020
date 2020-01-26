import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim

from agents.logging import Logger
from .dqn import DQNAgent


class OneHot(nn.Module):
    """ Encodes inputs input one hot vectors
    """

    def __init__(self, depth: int, dtype: torch.dtype = torch.float32):
        """

        Args:
            depth: The size of the one_hot dimension
            dtype: The data type of the output tensor
        """
        super().__init__()
        self.dept = depth
        self.dtype = dtype

    def forward(self, indices):
        x_one_hot = indices.new_zeros(tuple(indices.size()) + (self.dept,))
        x_one_hot.scatter_(dim=-1, index=torch.unsqueeze(indices, dim=-1), src=indices.new_ones(x_one_hot.size()))
        return x_one_hot.to(dtype=self.dtype)


class ConvNet(nn.Module):

    def __init__(self, obs_shape: Tuple[int, int, int], conv_sizes: List[Tuple[int, int, int, int]]) -> None:
        super().__init__()
        self.network = nn.Sequential()
        out_h, out_w, out_c = obs_shape
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_sizes):
            layer = nn.Conv2d(out_c, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.network.add_module(f'conv2d_{i + 1}', layer)
            self.network.add_module(f'ReLU_{i + 1}', nn.ReLU())
            out_h = ((out_h + 2 * padding - kernel_size) // stride) + 1
            out_w = ((out_w + 2 * padding - kernel_size) // stride) + 1
            out_c = out_channels
        self.out_features = out_h * out_w * out_c

    def forward(self, states: torch.Tensor):
        states = states.permute(0, 3, 1, 2)  # NxHxWxC -> NxCxHxW
        return self.network(states)


class DenseNet(nn.Module):

    def __init__(self, in_features: int, fc_sizes: List[int], out_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential()
        for i, size in enumerate(fc_sizes):
            self.network.add_module(f'fc_{i}', nn.Linear(in_features, size))
            self.network.add_module(f'ReLU_{i}', nn.ReLU())
            in_features = size
        self.network.add_module('fc_out', nn.Linear(in_features, out_features))
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor):
        return self.network(inputs)


class IntrinsicCuriosityModule(nn.Module):

    def __init__(self, obs_shape, num_actions, embed_convs: List[Tuple[int, int, int, int]] = None,
                 forward_hidden: List[int] = None, inverse_hidden: List[int] = None) -> None:
        super().__init__()
        embed_convs = embed_convs or [(32, 3, 2, 1), (32, 3, 1, 1), (32, 3, 1, 1), (32, 3, 1, 1)]
        forward_hidden = forward_hidden or [256]
        inverse_hidden = inverse_hidden or [256]
        self.num_actions = num_actions
        self.action_one_hot = OneHot(num_actions)
        self.state_embed = ConvNet(obs_shape, embed_convs)
        self.forward_model = DenseNet(num_actions + self.state_embed.out_features, forward_hidden,
                                      self.state_embed.out_features)
        self.inverse_model = DenseNet(2 * self.state_embed.out_features, inverse_hidden, num_actions)

    def forward(self, state, action, next_state):
        a_t_one_hot = self.action_one_hot(action)
        phi_t = torch.flatten(self.state_embed(state), start_dim=1)
        phi_tp1 = torch.flatten(self.state_embed(next_state), start_dim=1)
        forward = self.forward_model(torch.cat([a_t_one_hot, phi_t], dim=-1))
        inverse = self.inverse_model(torch.cat([phi_t, phi_tp1], dim=-1))
        return phi_tp1, inverse, forward


class CuriosityDQNAgent(DQNAgent):

    def __init__(self, env, dqn_factory, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                 target_update_interval, eta=0.01, beta=0.2, lmbda=0.1, lr=1e-3,
                 icm_embed_convs: List[Tuple[int, int, int, int]] = None, icm_forward_hidden: List[int] = None,
                 icm_inverse_hidden: List[int] = None, logger: Logger = None):
        # Initialize agent
        super().__init__(env=env, dqn_factory=dqn_factory, gamma=gamma, epsilon_start=epsilon_start,
                         epsilon_decay=epsilon_decay, epsilon_end=epsilon_end, memory_size=memory_size,
                         batch_size=batch_size, target_update_interval=target_update_interval, logger=logger)

        # Set the update interval for the target network
        self.target_update_interval = target_update_interval

        self.network_fn = dqn_factory
        self.eta = eta
        self.beta = beta
        self.lmbda = lmbda
        self.lr = lr

        self.icm_embed_convs = icm_embed_convs
        self.icm_forward_hidden = icm_forward_hidden
        self.icm_inverse_hidden = icm_inverse_hidden

    def create_qnetwork(self):
        # Create network
        network, _ = self.network_fn.create_qnetwork(target_qnetwork=False)

        self.icm = IntrinsicCuriosityModule(self.env.observation_space.shape, self.env.action_space.n,
                                            embed_convs=self.icm_embed_convs, forward_hidden=self.icm_forward_hidden,
                                            inverse_hidden=self.icm_inverse_hidden)

        model = nn.ModuleList([network, self.icm])

        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        return network, optimizer

    def reset(self):
        # Reset agent
        super().reset()

        # Create target network with episode counter
        self.target_qnetwork, _ = self.create_qnetwork()
        self.num_episode = 0
        self.episode_reward = 0
        self.total_steps = 0

    def act(self, state):
        # Exploration rate
        epsilon = 0.01 if self.is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.qnetwork([state])[0]
            return q_values.argmax().item()  # Greedy action

    def learn(self, state, action, reward, next_state, done):
        self.episode_reward += reward
        self.total_steps += 1
        if done:  # Increment episode counter at the end of each episode
            self.num_episode += 1
            self.logger.log_dict(self.total_steps, {
                'episode_reward': self.episode_reward,
                'memory_size': len(self.memory),
            })
            self.episode_reward = 0

        # Update target network with current one
        if self.num_episode % self.target_update_interval == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

        # Epsilon decay
        if done:
            self.epsilons.append(self.epsilon)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            # Sample batch of experience
            batch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)

            action = torch.LongTensor(action)
            reward = Tensor(reward)
            done = Tensor(done)

            if torch.cuda.is_available():
                action = action.cuda()
                reward = reward.cuda()
                done = done.cuda()

            # Q-value for current state given current action
            q_values = self.qnetwork(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_qnetwork(next_state)
            next_q_value = next_q_values.max(1)[0]

            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            dqn_loss = (q_value - td_target.detach()).pow(2).mean()

            state = Tensor(state)
            next_state = Tensor(next_state)

            if torch.cuda.is_available():
                state = state.cuda()
                next_state = next_state.cuda()

            # intrinsic reward
            phi_tp1, inverse, forward = self.icm(state, action, next_state)
            reward_intrinsic = self.eta * 0.5 * ((phi_tp1 - forward).pow(2)).sum(1).squeeze().detach()
            reward_extrinsic = reward
            reward = reward + reward_intrinsic

            # Intrinsic curiosity module loss
            inverse_loss = F.cross_entropy(inverse, action.detach())
            forward_loss = 0.5 * ((forward - phi_tp1).pow(2)).mean()
            icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

            loss = self.lmbda * dqn_loss + icm_loss

            self.optimizer.zero_grad()
            icm_loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'curiosity/loss': loss.data.cpu().numpy(),
                'curiosity/dqn_loss': dqn_loss.data.cpu().numpy(),
                'curiosity/inverse_loss': inverse_loss.data.cpu().numpy(),
                'curiosity/forward_loss': forward_loss.data.cpu().numpy(),
                'curiosity/icm_loss': icm_loss.data.cpu().numpy(),
                'curiosity/reward': reward.mean().data.cpu().numpy(),
                'curiosity/reward_extrinsic': reward_extrinsic.mean().data.cpu().numpy(),
                'curiosity/reward_intrinsic': reward_intrinsic.mean().data.cpu().numpy(),
            })
