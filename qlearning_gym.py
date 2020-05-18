## Open AI Gym - Q Learning
import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import sys

select = sys.argv[1]
#select = 'bot' or 'car'

if select == 'car':
    #Mountain Car parameters
    env = gym.make('MountainCar-v0')
    Q_table = np.zeros((20,20,3))
    buckets=[20, 20]
    discount=0.99
    exploration_max=0.03
    exploration_min=0.001
    learning_rate=0.4
    epochs=5000
else:
    #Acrobot parameters
    env = gym.make('Acrobot-v1')
    Q_table = np.zeros((2,2,2,2,8,16,3))
    buckets=[2,2,2,2,8,16]
    discount=0.99
    exploration_max=0.03
    exploration_min=0.001
    learning_rate=0.4
    epochs=5000

print(env.observation_space.high)
print(env.observation_space.low)


def toDiscreteStates(observation):
    interval=[0 for i in range(len(observation))]
    if select=='car':
        max_range=[1.2,0.07] #Mountain Car parameters
    else:
        max_range=[1,1,1,1,13,29] #Acrobot parameters

    for i in range(len(observation)):
        data = observation[i]
        inter = int(math.floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
        if inter>=buckets[i]:
            interval[i]=buckets[i]-1
        elif inter<0:
            interval[i]=0
        else:
            interval[i]=inter
    return interval

def get_action(observation,episode):
    exploration_prob = max(exploration_min, exploration_max / (int(episode/100) + 1))
    if episode%1000 == 0:
        print("exploration_prob", exploration_prob)

    if np.random.random() < exploration_prob:
        return env.action_space.sample()

    interval = toDiscreteStates(observation)
    return np.argmax(np.array(Q_table[tuple(interval)]))

def updateQ(next_observation,reward,action,current_observation,t):
    next_interval = toDiscreteStates(next_observation)
    Q_next = max(Q_table[tuple(next_interval)])

    current_interval = toDiscreteStates(current_observation)
    Q_table[tuple(current_interval)][action] += learning_rate * (reward + discount*(Q_next) - Q_table[tuple(current_interval)][action])

rewards=[]
episodes_rewards = []
for episode in range(epochs):
    observation = env.reset()
    episode_reward=0
    t=0
    while (True):
        t+=1
        #env.render()

        action = get_action(observation,episode)
        next_observation, reward, done, info = env.step(action)
        episode_reward += reward

        updateQ(next_observation,reward,action,observation,t)
        observation=next_observation

        if done:
            print("Episode",episode,"finished after",t,"timesteps")
            episodes_rewards.append(episode_reward)
            if episode%50 == 0:
                avg_reward = mean(episodes_rewards)
                rewards.append(avg_reward)
                episodes_rewards = []
            #print('i',episode,'reward',episode_reward)
            break

# plot rewards
plt.plot(np.arange(0,epochs,50),rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward vs episodes")
plt.show()

plt.plot(np.arange(0,epochs,50),np.negative(rewards))
plt.xlabel("Episode")
plt.ylabel("Time steps")
plt.title("Time steps vs episodes")
plt.show()

#visualize in Gym
observation = env.reset()
episode_reward=0
t=0
while (True):
    env.render()
    action = get_action(observation,episode)
    observation, reward, done, info = env.step(action)
    episode_reward += reward
    t+=1
    if done:
        print("Episode finished after",t,"timesteps")
        print('reward',episode_reward)
        env.close()
        break