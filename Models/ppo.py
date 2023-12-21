import numpy as np
from gymnasium.spaces import flatdim

from loggers import Logger
from Models import MLP

import torch
import torch.nn.functional as func
from gymnasium import Env
from torch import optim


class PPO:
    hyperparameters = {
        "max_time_steps_per_batch": 8_000,
        "max_time_steps_per_episode": 2_000,
        "updates_per_iteration": 5,
        "save_freq": 10,
        "save_path": "SavedModels/PPO",
        "lr": 3e-3,
        "gamma": 0.99,
        "clip": 0.2
    }

    def __init__(self, env: Env, layers: list[int], **updated_hyperparameters):
        self.logger = Logger(print_order=["Average_rewards", "Average_lens"], print_every=1,
                             wandb_details={"project": "RocketSim", "config": None, "log_keys": ["Average_rewards"]})

        self.env = env
        self.obs_dims = flatdim(env.observation_space)
        self.act_dims = flatdim(env.action_space)

        self.max_time_steps_per_batch = None
        self.max_time_steps_per_episode = None
        self.updates_per_iteration = None

        self.save_freq = None
        self.save_path = None

        self.lr = None
        self.gamma = None
        self.clip = None

        self.hyperparameters.update(updated_hyperparameters)

        for param, val in self.hyperparameters.items():
            if type(val) == str:
                val = f"\"{val}\""
            exec("self." + param + " = " + str(val))

        if self.lr is None:
            self.lr = 1e-3

        self.actor = MLP(self.obs_dims, self.act_dims, layers, final_activation_function=[torch.nn.Softmax(dim=0)])
        self.critic = MLP(self.obs_dims, 1, layers)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_tensor = torch.full(size=(self.act_dims,), fill_value=0.5)
        self.covariance_matrix = torch.diag(self.covariance_tensor)

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance_matrix)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

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

            obs, _ = self.env.reset()

            ep_t = 0
            for ep_t in range(self.max_time_steps_per_episode):
                total_time_step += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, terminated, _, _ = self.env.step(torch.argmax(action).item())

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
        self.actor.save(self.save_path, "ppo_actor")
        self.critic.save(self.save_path, "ppo_critic")
