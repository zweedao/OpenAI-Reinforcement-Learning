## Open AI Gym Mountain Car - Reinforce
import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import torch
from torch import nn
from torch import optim
import random
from gym import wrappers
from joblib import dump, load

folder_url = "./"
model_url = folder_url + 'models/'

#Policy neutal network
class policy_net(nn.Module):
    def __init__(self, env):
        super(policy_net, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, 20),
            nn.ReLU(),
            #nn.Linear(20, 20),
            #nn.ReLU(),
            nn.Linear(20, self.n_outputs),
            nn.Softmax(dim=-1))

    def forward(self, state):
        state = torch.FloatTensor(state)
        action_prob = self.model(state)
        return action_prob

def get_good_games(env, epochs, discount):
    # Set up lists to hold results
    total_rewards = []
    avg_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    total_losses = []
    avg_losses = []
    batch_counter = 0

    action_space = np.arange(env.action_space.n)

    for episode in range(epochs):
        state = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while not done:
            action = random.randrange(0, 3)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = next_state

            # If done, batch data
            if done:
                print("Good games:",batch_counter,"Episode:",episode+1,"Reward:",sum(rewards))
                #for Mountain Car only
                if env_name == 'MountainCar-v0':
                    if sum(rewards) < -999:
                        break

                batch_rewards.extend(discount_norm_rewards(rewards, discount))
                batch_states.extend(states)
                batch_actions.extend(actions)
                total_rewards.append(sum(rewards))
                batch_counter += 1

    return batch_states, batch_actions, batch_rewards, total_rewards


def train_policy(policy_model, epochs, learning_rate, batch_states, batch_actions, batch_rewards):
    # Define optimizer & loss function
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    #tensor
    onehot_actions = np.zeros((len(batch_actions), env.action_space.n))
    onehot_actions[np.arange(len(batch_actions)), batch_actions] = 1

    state_tensor = torch.FloatTensor(batch_states)
    reward_tensor = torch.FloatTensor(batch_rewards)
    action_tensor = torch.FloatTensor(onehot_actions)

    #training
    total_losses = []

    for i in range(epochs):
        input = state_tensor
        target = action_tensor

        optimizer.zero_grad()

        # Calculate loss
        output = policy_model(input)
        loss = loss_function(output, target)

        total_losses.append(loss.item())

        # Calculate gradients
        loss.backward()
        # Apply gradients
        optimizer.step()

        print("epoch:",i,"Loss:",loss.item())

    #plot loss
    plt.plot(total_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.show()

    return policy_model


## Environment: Mountain Car
env_name = 'MountainCar-v0'
env = gym.make(env_name)
env._max_episode_steps = 1000

#Hyper parameters
discount = 0.99
epochs=100
learning_rate=0.01

#Play lots of games
batch_states, batch_actions, batch_rewards, total_rewards = get_good_games(env, 100000, discount)

#Save good games
dump(batch_states, model_url + 'batch_states.joblib')
dump(batch_actions, model_url + 'batch_actions.joblib')
dump(batch_rewards, model_url + 'batch_rewards.joblib')
dump(total_rewards, model_url + 'total_rewards.joblib')

#Load good games
batch_states = load(model_url + 'batch_states.joblib')
batch_actions = load(model_url + 'batch_actions.joblib')
batch_rewards = load(model_url + 'batch_rewards.joblib')

#train policy model
policy_model = policy_net(env)
policy_model = train_policy(policy_model, epochs, learning_rate, batch_states, batch_actions, batch_rewards)

#visualize result in gym
observation = env.reset()
episode_reward=0
t=0
while (True):
    env.render()
    action_prob = policy_model(observation).detach().numpy()
    action = np.argmax(action_prob)
    observation, reward, done, info = env.step(action)
    episode_reward += reward
    t+=1
    if done:
        print("Episode finished after",t,"timesteps")
        env.close()
        break