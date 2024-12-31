import itertools
import os
import random

import matplotlib.pyplot as plt
import torch
from torch import inference_mode
from torch.optim import AdamW

from alphadave.core.model import DQN
from alphadave.core.replay_buffer import PrioritizedReplayBuffer
from alphadave.core.scheduler import EpsilonScheduler
from alphadave.utils import get_logger, load_checkpoint, save_checkpoint
from alphadave.wrapper import TorchEnv


class AlphaDave:

    def __init__(self, env: TorchEnv):

        self.env = env
        self.policy_dqn = DQN(in_states=env.num_states, out_actions=env.num_actions)
        self.target_dqn = DQN(in_states=env.num_states, out_actions=env.num_actions)

        self.optimizer = None
        self.replay_buffer = None
        self.epsilon_scheduler = None

    def train(
        self,
        seed,
        batch_size,
        learning_rate,
        replay_buffer_capacity,
        replay_buffer_alpha,
        replay_buffer_beta,
        network_sync_interval,
        epsilon_start,
        epsilon_final,
        epsilon_decay,
        gamma,
        save_dir,
        save_interval,
        stop_at_episode=None,
        from_checkpoint=None,
    ):

        # initialization
        self.optimizer = AdamW(self.policy_dqn.parameters(), learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_capacity, replay_buffer_alpha, replay_buffer_beta)
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_final, epsilon_decay)

        # track variables
        start_episode = 1
        losses_per_episode = []
        rewards_per_episode = []

        # get logger
        logger = get_logger(save_dir)

        # load checkpoint
        if from_checkpoint:
            logger.info(f'===== Resume from checkpoint {from_checkpoint} =====')
            ckpt = load_checkpoint(from_checkpoint)
            if stop_at_episode != None and ckpt['episode'] >= stop_at_episode:
                logger.info(f'===== Stop at episode {stop_at_episode} =====')
                exit()
            self.policy_dqn.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.replay_buffer = ckpt['replay_buffer']
            start_episode = ckpt['episode'] + 1
            losses_per_episode = ckpt['losses_per_episode']
            rewards_per_episode = ckpt['rewards_per_episode']

        # prepare the policy and target network
        self.policy_dqn.train()
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # train indefinitely
        for i in itertools.count(start_episode):

            state, _ = self.env.reset(seed=seed)
            epsilon = self.epsilon_scheduler(i)
            terminated = False
            total_reward = 0

            while not terminated:

                # get action mask
                mask = self.env.unwrapped.get_action_mask()

                # select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = self.env.action_space.sample(mask)
                else:
                    # select best action
                    with torch.no_grad():
                        action = self.policy_dqn(state, mask).argmax().item()

                # execute action
                next_state, reward, terminated, _, info = self.env.step(action)

                # extract info
                is_win = info['is_win']
                next_mask = info['action_mask']

                # update rewards
                total_reward += reward

                # append experience into replay memory
                sunbean_id = 1  # 排除阳光豆的action（减少模型的影响因素）
                if action == 0 or self.env.unwrapped.action_to_plant_id(action) != sunbean_id:
                    self.replay_buffer.append((state, mask, action, reward, terminated, next_state, next_mask))

                # move to the next state
                state = next_state

            # track rewards per episode
            rewards_per_episode.append(total_reward)

            # enough experience has been collected
            if len(self.replay_buffer) > batch_size:
                # get mini batch
                mini_batch = self.replay_buffer.sample(batch_size)
                # optimize
                loss = self.optimize(mini_batch, gamma)
                losses_per_episode.append(loss)
                logger.info(
                    f'Episode {i}, '
                    f'rewards: {total_reward:.4f}, '
                    f'loss: {loss:.4f}, '
                    f'epsilon: {epsilon:.4f}, '
                    f'memory_size: {len(self.replay_buffer)}, '
                    f'is_win: {is_win}'
                )

            # sync target network
            if i % network_sync_interval == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            # save policy
            if save_interval != None and i % save_interval == 0:
                ckpt_to_save = {
                    'model_state_dict': self.policy_dqn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'replay_buffer': self.replay_buffer,
                    'episode': i,
                    'losses_per_episode': losses_per_episode,
                    'rewards_per_episode': rewards_per_episode,
                }
                save_checkpoint(ckpt_to_save, save_dir)

            # stop at specified episode
            if stop_at_episode != None and i >= stop_at_episode:
                break

        # save figure
        self.save_figure(save_dir, rewards_per_episode, losses_per_episode)

        # close environment
        self.env.close()

    def optimize(self, mini_batch, gamma):
        self.optimizer.zero_grad()

        samples, indices, weights = mini_batch
        states, masks, actions, rewards, terminations, next_states, next_masks = zip(*samples)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards).float()
        terminations = torch.tensor(terminations).float()
        next_states = torch.stack(next_states)
        weights = torch.tensor(weights)

        output_q = self.policy_dqn(states, masks).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.policy_dqn(next_states, next_masks).argmax(dim=1)
            next_q = self.target_dqn(next_states, next_masks).gather(dim=1, index=next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - terminations) * gamma * next_q

        weighted_error = (output_q - target_q).pow(2) * weights
        loss = weighted_error.mean()
        loss.backward()

        self.replay_buffer.set_priorities(indices, weighted_error.detach().numpy())
        self.optimizer.step()

        return loss.item()

    @inference_mode
    def play(self):
        self.policy_dqn.eval()
        state, _ = self.env.reset()
        terminated = False
        while not terminated:
            mask = self.env.unwrapped.get_action_mask()
            action = self.policy_dqn(state, mask).argmax().item()
            next_state, _, terminated, _, _ = self.env.step(action)
            state = next_state
        self.env.close()

    @staticmethod
    def save_figure(save_dir, rewards, losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title(f'Reward')
        plt.plot(rewards)
        plt.subplot(122)
        plt.title('Loss')
        plt.plot(losses)
        plt.savefig(os.path.join(save_dir, 'figure.png'))
