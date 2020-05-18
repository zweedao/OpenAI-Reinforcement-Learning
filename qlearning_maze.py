## Q learning
import sys
folder_url = "./"
sys.path.append(folder_url)
from evaluation import *
from maze import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def get_Q(state, action, Q_dict):
    if (state, action) not in Q_dict:
        Q_dict[(state, action)] = 0
    return Q_dict[(state, action)]

def get_V(state, action_list, Q_dict):
    V = float('-inf')
    best_action = None

    for action in action_list:
        Q = get_Q(state, action, Q_dict)
        V = max(V, Q)
        if Q >= V:
            best_action = action

    return V, best_action

#Hyper parameters
discount=0.9
exploration=0.5
learning_rate=0.2
epochs = 5000

#Intitialize
env = Maze()

Q_values = np.zeros((env.snum, env.anum))
Q_dict = {} #dictionary to store Q values, for faster access
Q_optimal = np.load(folder_url + 'results/Q_values.npy', allow_pickle=True)

action_list = np.arange(env.anum).tolist()
eval_steps, eval_reward, RMSE = [], [], []

# Q-learning
for i in range(epochs):
    #i=0 #debug
    print('i',i)
    state = env.reset()
    done = False
    while not done:
        if np.random.random() < exploration:
            best_action = np.random.choice(action_list)
        else:
            _, best_action = get_V(state, action_list, Q_dict)

        #print('action best', best_action)
        reward, next_state, done = env.step(state, best_action, Q_dict)
        V_next, _ = get_V(next_state, action_list, Q_dict)
        Q_dict[(state, best_action)] = (1-learning_rate) * get_Q(state, best_action, Q_dict) + learning_rate * (reward + discount*V_next)
        state = next_state

    #populate Q values table
    for k, v in Q_dict.items():
        Q_values[k] = v

    avg_step, avg_reward = evaluation(env, Q_values)
    eval_steps.append(avg_step)
    eval_reward.append(avg_reward)
    RMSE.append(np.sqrt(np.mean((Q_values-Q_optimal)**2)))


#Plot
fig1 = plt.figure()
plt.plot(eval_steps)
plt.xlabel("Epochs")
plt.ylabel("Steps")
plt.title("Steps vs epochs")
plt.show(fig1)

fig2 = plt.figure()
plt.plot(eval_reward)
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title("Reward vs epochs")
plt.show(fig2)

fig3 = plt.figure()
plt.plot(RMSE)
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("RMSE vs epochs")
plt.show(fig3)