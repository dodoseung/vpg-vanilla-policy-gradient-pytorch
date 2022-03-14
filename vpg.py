# Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
from collections import deque

class VPGNet(nn.Module):
    def __init__(self, input, output):
        super(VPGNet, self).__init__()
        self.input = nn.Linear(input, 16)
        self.fc = nn.Linear(16, 16)
        self.output = nn.Linear(16, output)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        x = F.softmax(self.output(x))
        return x
    
class VPG():
    def __init__(self, env, gamma=0.95, learning_rate=1e-3):
        super(VPG, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
   
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vanilla policy gradient network
        self.policy_net = VPGNet(self.state_num, self.action_num).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Learning setting
        self.gamma = gamma
        
    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        policy = self.policy_net(state).cpu().detach().numpy()
        action = np.random.choice(self.action_num, 1, p=policy[0])
        return action[0]

    # Learn the policy
    # G: Expected reward
    # j: Policy objective function
    def learn(self, states, actions, rewards):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Calculate the values
        for i in reversed(range(len(rewards)-1)):
            rewards[i] += self.gamma * rewards[i+1]
        G = torch.FloatTensor(rewards).to(self.device)

        # Calculate objective function
        log_prob = torch.log(self.policy_net(states))
        j = G * log_prob[range(len(actions)), actions]
        
        # Calculate the loss and optimize the network
        loss = -j.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    ep_rewards = deque(maxlen=100)
    
    env = gym.make("CartPole-v0")
    agent = VPG(env, gamma=0.99, learning_rate=1e-3)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        states = []
        actions = []
        rewards = []

        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                ep_rewards.append(sum(rewards))
                agent.learn(states, actions, rewards)
                
                if i % 100 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()