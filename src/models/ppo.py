import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.models.gae import gae

class PPO(object):
    def __init__(self, model, optimizer, device, env):
        self.model = model.to(device)
        self.optimizer = optimizer.to(device)
        self.device = device
        self.env = env

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                           rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions,
                                                                                  log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            action = dist.sample()
            torch.clamp(action, -self.model.max_output_value, self.model.max_output_value)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            total_reward += total_reward
        return total_reward


    def train(self, max_frames, num_steps, ):
        frame_idx = 0
        early_stop = False
        state = self.env.reset()
        test_rewards = []
        while frame_idx < max_frames and not early_stop:

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.model(state)

                action = dist.sample()
                torch.clamp(action, -self.model.max_output_value, self.model.max_output_value)
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                if done:
                    next_state = self.env.reset()
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)

                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test() for _ in range(10)])
                    test_rewards.append(test_reward)
                    print('test reward: ', test_reward)

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
