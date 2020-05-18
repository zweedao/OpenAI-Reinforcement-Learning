# Reinforcemnent Learning

Using various Reinforcement Learning algorithms to solve the following 3 problems. Full details can be found in `Project Report.pdf`.

### 1. Maze Navigation

How to get from S to G in the Maze, and collect the most number of flags F along the way.

<img src="images/maze.png" width="200px">

### 2. [Open AI Acrobot](https://gym.openai.com/envs/Acrobot-v1/) ([See video result](https://youtu.be/ecF-w1arABs))

How to swing the bot’s arm to reach a certain height.

<img src="images/acrobot.png" width="200px">

### 3. [Open AI Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) ([See video result](https://youtu.be/weExxc8X95Q))

How to push & pull the car to make it reach the mountain’s top.

<img src="images/car.png" width="250px">

## How to run

Run these commands in your Terminal:

### 1. Maze Navigation problem:

#### 1.1 Value iteration algorithm:

    python value_iteration.py

#### 1.2 Q-learning algorithm: 

    python qlearning_maze.py

### 2. Open AI Acrobot problem:

#### 2.1 Deep reinforcement learning - REINFORCE algorithm with Neural network:

    python reinforce_bot.py

#### 2.2 Q-learning algorithm:

    python qlearning_gym.py bot


### 3. Open AI Mountain Car problem:

#### 3.1 Deep reinforcement learning - REINFORCE algorithm with Neural network:

    python reinforce_car.py

#### 3.2 Q-learning algorithm:

    python qlearning_gym.py car
