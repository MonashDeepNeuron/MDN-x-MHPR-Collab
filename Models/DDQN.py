import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from env_wrapper import register_env
from reward_functions import RewardFunction, RewardFunctions

class Linear_DDQN(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size, lr):
        super().__init__()
        self.linear1 = nn.Linear(*input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions
    
    def save(self, file_name='model_presentation.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    

class TrainingAgent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, output_size,
                 max_mem_size=100000, eps_end=0.01, eps_dec=3e-4, tau=0.01, update_target_every=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(output_size)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        #NEW STUFF
        self.tau = tau
        self.update_target_every = update_target_every
        self.target_update_counter = 0

        self.Q_eval = Linear_DDQN(input_size=input_size, output_size = output_size, lr = self.lr, hidden_size1=256, hidden_size2 = 256)
        self.Q_target = Linear_DDQN(input_size=input_size, output_size = output_size, lr = self.lr, hidden_size1=256, hidden_size2 = 256)

        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()
        #torch.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)  # You can adjust the max_norm value as needed

        self.state_memory = np.zeros((self.mem_size, *input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_size), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def update_target_network_soft(self):
        for target_param, eval_param in zip(self.Q_target.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def update_target_network_hard(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([np.array(observation)]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next_actions = torch.argmax(self.Q_eval.forward(new_state_batch), dim=1)
        q_next = self.Q_target.forward(new_state_batch)  # Use target network for Q-value evaluation
        q_next = q_next[batch_index, q_next_actions]  # Select Q-values based on the selected actions

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0) 
        
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.update_target_network_hard()  # Perform a hard update periodically
        else:
            self.update_target_network_soft()  # Otherwise, perform a soft update
            
class TestingAgent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, output_size,
                 max_mem_size=100000, eps_end=0.01, eps_dec=3e-4, tau=0.01, update_target_every=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(output_size)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        #NEW STUFF
        self.tau = tau
        self.update_target_every = update_target_every
        self.target_update_counter = 0

        file_name_model = "./model/DDQN_Rocket.pth"

        self.Q_eval = Linear_DDQN(input_size=input_size, output_size = output_size, lr = self.lr, hidden_size1=256, hidden_size2 = 256)
        self.Q_eval.load_state_dict(torch.load(file_name_model))

    def choose_action(self, observation):
        state = torch.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = torch.argmax(actions).item()
    
        return action

class Train():
    def __init__(self,agent):
        self.agent = agent
        # Agent will be: 
        # TrainingAgent(gamma=0.99, epsilon=1.0, batch_size=64, output_size=4, eps_end=0.01,input_size=[1], lr=5e-4)
    
    def train(self,n_games):

        register_env()
        env = gym.make("RocketSim-v0", reward_function=RewardFunction(RewardFunctions.TestReward))

        #env = gym.make("LunarLander-v2")
        agent = self.agent
        scores, eps_history = [], []
        high_score = 0
    
        for i in range(n_games):
            score = 0
            done = False
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, _, _= env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
            scores.append(score)

            if i > 60 and score > high_score:
                agent.Q_eval.save("rocket_sim.pth")

            if score > high_score:
                high_score = score
        
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])

            print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        x = [i+1 for i in range(n_games)]
    
