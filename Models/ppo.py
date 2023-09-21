import os

import numpy as np
from gymnasium.spaces import flatdim

from loggers import Logger
from Models import MLP

import torch
import torch.nn.functional as func
from gymnasium import Env
from torch import optim


class PPO:
    hyperparams = {

    }

    def __init__(self, env: Env, layers: list[int], save_path="models/ppo", **hyperparameters):
        self.logger = Logger(print_order=["Average_rewards", "Average_lens"], print_every=5)

        self.env = env
        self.env_seed = 0
        self.obs_dims = flatdim(env.observation_space)
        self.act_dims = flatdim(env.action_space)

        self.max_time_steps_per_batch = 8_000
        self.max_time_steps_per_episode = 2_000
        self.updates_per_iteration = 5

        self.save_freq = 10
        self.save_path = save_path + f"/{env.unwrapped.spec.id}/"

        self.lr = 3e-3
        self.gamma = 0.95
        self.clip = 0.2

        self.actor = MLP(self.obs_dims, self.act_dims, layers)
        self.critic = MLP(self.obs_dims, 1, layers)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_tensor = torch.full(size=(self.act_dims,), fill_value=0.5)
        self.covariance_matrix = torch.diag(self.covariance_tensor)

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance_matrix)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []
        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []

        total_time_step = 0

        while total_time_step < self.max_time_steps_per_batch:
            ep_rewards = []

            obs, _ = self.env.reset(seed=self.env_seed)
            self.env_seed += 1

            ep_t = 0
            for ep_t in range(self.max_time_steps_per_episode):
                total_time_step += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, terminated, _, _ = self.env.step(action)

                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if terminated:
                    break

            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rewards)

        self.logger["Average_rewards"] = round(float(np.mean([np.sum(eps_rewards) for eps_rewards in batch_rewards])),
                                               3)
        self.logger["Average_lens"] = round(float(np.mean(batch_lens)), 3)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        v = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance_matrix)
        log_prob = dist.log_prob(batch_acts)

        return v, log_prob

    def learn(self, total_time_steps):
        time_step = 0
        iteration_count = 0
        self.logger.total_steps = total_time_steps

        while time_step < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            time_step += sum(batch_lens)
            iteration_count += 1

            v, _ = self.evaluate(batch_obs, batch_acts)
            a_k = batch_rtgs - v.detach()
            a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10)

            for _ in range(self.updates_per_iteration):
                v, log_prob = self.evaluate(batch_obs, batch_acts)
                ratio = torch.exp(log_prob - batch_log_probs)
                surrogate1 = ratio * a_k
                surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_k

                actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
                critic_loss = func.mse_loss(v, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            self.logger.step()

            if iteration_count % self.save_freq == 0:
                self.save_model()
        self.save_model()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        compiled = "_compiled" if self.actor.is_compiled else ""
        torch.save(self.actor.state_dict(), self.save_path + f"ppo_actor{compiled}.pth")
        torch.save(self.critic.state_dict(), self.save_path + f"ppo_critic{compiled}.pth")
