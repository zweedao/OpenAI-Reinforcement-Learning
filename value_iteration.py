import sys
folder_url = "./"
sys.path.append(folder_url)
from evaluation import *
from maze import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


## Value Iteration
def get_Q(state, action, V_values):
    Q = 0
    reward_list = []

    slip_action = ACTMAP[action]
    slip_reward, slip_next_state, _ = env.step(state, slip_action, slip_chance=False)
    reward_list.append((slip_reward, slip_next_state, env.slip))

    reward, next_state, _ = env.step(state, action, slip_chance=False)
    reward_list.append((reward, next_state, 1-env.slip))

    for reward, next_state, probability in reward_list:
        Q += probability * (reward + discount * V_values[next_state])
    return Q

#Hyper parameters
discount=0.9
epochs=1000

#Intinialize
env = Maze()
V_values = np.zeros(env.snum)
Q_values = np.zeros((env.snum, env.anum))
optimal_policy = np.zeros(env.snum)

# Value iteration
for i in range(epochs):
    print('i',i)
    for state in range(env.snum):
        if env.idx2cell[int(state/8)] == env.goal_pos:
            continue
        V = float('-inf')
        for action in range(env.anum):
            Q = get_Q(state, action, V_values)
            V = max(V, Q)
        V_values[state] = V

for state in range(env.snum):
    for action in range(env.anum):
        Q_values[state, action] = get_Q(state, action, V_values)

for state in range(env.snum):
    best_action = np.argmax(Q_values[state,:])
    optimal_policy[state] = best_action

#save Q_values
np.save(folder_url + 'results/Q_values',Q_values)

#plot the path on map
state = 0
done = False
while not done:
    action = int(optimal_policy[state])
    reward, next_state, done = env.step(state, action, slip_chance=False)
    print('state:',state,'action:',["UP","DOWN","LEFT","RIGHT"][action],'reward:',reward)
    env.plot(state, action)
    state = next_state
print('final state:',state,'reward:',reward)
env.plot(state, action)