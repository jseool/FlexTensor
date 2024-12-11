import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from threading import Thread
from queue import Queue
import time

class A3CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CNetwork, self).__init__()
        self.shared_layer = nn.Linear(state_dim, 128)
        self.policy_layer = nn.Linear(128, action_dim)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.shared_layer(state))
        logits = self.policy_layer(x)
        value = self.value_layer(x)
        return logits, value

class A3CWorker(Thread):
    def __init__(self, worker_id, global_model, optimizer, state_space, configs, gamma=0.99, update_interval=5):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = A3CNetwork(state_space.state_dim, state_space.action_dim)
        self.local_model.load_state_dict(global_model.state_dict())  # Sync with global model
        self.state_space = state_space
        self.gamma = gamma
        self.update_interval = update_interval
        self.configs = configs
        self.done = False

    def run(self):
        states, actions, rewards, values = [], [], [], []
        trial = 0
        while not self.done and trial < self.configs.trial:
            state = self.state_space.reset()
            total_reward = 0
            for _ in range(self.update_interval):
                states_np = np.array(states)  
                states_tensor = torch.FloatTensor(states_np)
                logits, value = self.local_model(states_tensor)
                # logits, value = self.local_model(torch.FloatTensor(state))
                action_prob = torch.softmax(logits, dim=-1)
                action = np.random.choice(range(action_prob.shape[0]), p=action_prob.detach().numpy())
                next_state, reward, done, _ = self.state_space.step(action)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)

                state = next_state
                total_reward += reward

                if done:
                    break

            # Compute advantage and update
            if len(states) > 0:
                self._update_global_model(states, actions, rewards, values)

            trial += 1

    def _update_global_model(self, states, actions, rewards, values):
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values = torch.cat(values)

        # Compute advantages
        advantages = returns - values

        # Compute loss
        policy_loss = []
        value_loss = []
        for logit, value, action, advantage in zip(
            self.local_model(torch.FloatTensor(states))[0],
            values,
            actions,
            advantages,
        ):
            action_prob = torch.softmax(logit, dim=-1)
            log_prob = torch.log(action_prob[action])
            policy_loss.append(-log_prob * advantage.detach())
            value_loss.append((advantage ** 2))

        policy_loss = torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()
        total_loss = policy_loss + value_loss

        # Backpropagation and update global model
        self.optimizer.zero_grad()
        total_loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            global_param.grad = local_param.grad
        self.optimizer.step()

        # Sync local model with global model
        self.local_model.load_state_dict(self.global_model.state_dict())


class A3CStateSpace:
    def __init__(self, space):
        self.space = space
        self.state_dim = space.dim
        space.get_state_vector(space.get_initial_state_indices()).shape[0]
        self.action_dim = space.action_dim
        self.current_state_indices = None

    def reset(self):
        self.current_state_indices = self.space.get_initial_state_indices()
        state_vec = self.space.get_state_vector(self.current_state_indices)
        return state_vec

    def step(self, action):
        # use current_state_indices and action to get next state
        next_state_indices, reward, done = self.space.take_action(self.current_state_indices, action)
        self.current_state_indices = next_state_indices
        next_state_vec = self.space.get_state_vector(self.current_state_indices)
        # return next_state, reward, done, info(dict)
        return next_state_vec, reward, done, {}