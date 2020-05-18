## Open AI Gym Acrobot - Reinforce
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
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, self.n_outputs),
            nn.Softmax(dim=-1))

    def forward(self, state):
        state = torch.FloatTensor(state)
        action_prob = self.model(state)
        return action_prob

#discount reward
def discount_norm_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0
    for t in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma +episode_rewards[t]
        discounted_episode_rewards[t] = cumulative

    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return discounted_episode_rewards

#reinforce algorithm
def reinforce(env, policy_model, epochs, learning_rate, batch_size, discount):
    #debug
    #policy_model = policy_model
    #discount=0.99
    #epochs=100000
    #batch_size=1
    #learning_rate=0.1
    #exploration_max=0.5
    #exploration_min=0.001

    # Set up lists to hold results
    total_rewards = []
    avg_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    total_losses = []
    avg_losses = []
    batch_counter = 0

    # Define optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(policy_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)

    action_space = np.arange(env.action_space.n)

    for episode in range(epochs):
        state = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while not done:
            # Use policy_model to predict actions, play the game, record reward
            action_prob = policy_model(state).detach().numpy()
            action = np.random.choice(action_space, p=action_prob)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = next_state

            # If done, batch data
            if done:
                batch_rewards.extend(discount_norm_rewards(rewards, discount))
                batch_states.extend(states)
                batch_actions.extend(actions)
                total_rewards.append(sum(rewards))
                batch_counter += 1

                # If batch is done, update network
                if batch_counter % batch_size == 0 or episode == epochs-1:
                    optimizer.zero_grad()

                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    action_logprob = torch.log(policy_model(state_tensor))
                    expected_rewards = reward_tensor * action_logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -expected_rewards.mean()
                    #loss = expected_rewards.mean()
                    total_losses.append(loss.item())

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []

                    #print("Loss:",loss.item())

                # Print reward
                #if batch_counter % batch_size == 0:
                    avg_reward = np.mean(total_rewards[-batch_size:])
                    avg_rewards.append(avg_reward)
                    avg_loss = np.mean(total_losses[-batch_size:])
                    avg_losses.append(avg_loss)
                    print("Episode:",episode+1,
                          "Average reward:",avg_reward,"Average loss:", avg_loss)

    #plot rewards
    plt.plot(np.arange(0,epochs,batch_size),avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

    #plot time step
    plt.plot(np.arange(0,epochs,batch_size),np.negative(avg_rewards))
    plt.xlabel("Episode")
    plt.ylabel("Time step")
    plt.title("Time step vs Episode")
    plt.show()

    #plot loss
    plt.plot(np.arange(0,epochs,batch_size), avg_losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss vs Episode")
    plt.show()

    return policy_model


#Environment: Acrobot
#env_name = 'CartPole-v0'
env_name = 'Acrobot-v1'
env = gym.make(env_name)

#Hyper parameters
discount=0.99
epochs=3000
batch_size=50
learning_rate=0.01

#Train policy model
policy_model = policy_net(env)
policy_model = reinforce(env, policy_model, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, discount=discount)
dump(policy_model, model_url + 'reinforce_bot_model.joblib')