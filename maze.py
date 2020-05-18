# Heejin Chloe Jeong
# Additional comments added by Travers Rhodes

# Description:
# There are total 112 states defined by a position and a flag state.
# There are 8 possible "flag states" (indexed as given by num2flag)
# There are 13 possible "positions" (indexed as given by idx2cell)
# The overall state index is equal to positionIndex * 8 + flagStateIndex
# There are four cardinal actions.
# A reward will be given as equivalent to the number of flags you have collected at the goal state 
# (i.e. at the current state s, it performs an action a and observes a reward r and the next state s'. 
# If s'=goal state, r=the number of flags it has collected. Otherwise, r=0 ). 
# There are also six obstaces and the agent stays at the current state if it performs an action toward 
# an obstacle or off the map. The agent slips with a probability 0.1 and reaches the next clockwise 
# destination(i.e. It performed UP, but moved to its RIGHT).

import numpy as np
import random 
import pdb
# action ID : 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
# ACTMAP gives our definition of "clockwise slipping".
# That is, we meant to go 0:UP, but instead we go 3:Right. Etc.
ACTMAP = {0:3, 1:2, 2:0, 3:1}
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

class Maze():
    # state ID : 0, ..., 111
    # action ID : 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    obstacles = [(0,1),(0,3),(2,0),(2,4),(3,2),(3,4)]
    def __init__(self):
        self.episodic = True
        self.stochastic = True
        self.snum = 112
        self.anum = 4
        self.slip = 0.1
        self.dim = (4,5)
        self.start_pos = (0,0)
        self.goal_pos = (0,4)
        self.goal = (96,104)
        # self.map = np.asarray(["SWFWG","OOOOO","WOOOW","FOWFW"], dtype='c')
        self.map = np.asarray(["SWFWG","OOOOO","WOOOW","FOWFW"])
        self.img_map = np.ones(self.dim)
        for x in Maze.obstacles:
            self.img_map[x[0]][x[1]] = 0
        self.idx2cell = {0: (0, 0), 1: (1, 0), 2: (3, 0), 3: (1, 1), 4: (2, 1), 5: (3, 1),
            6: (0, 2), 7: (1, 2), 8: (2, 2), 9: (1, 3), 10: (2, 3), 11: (3, 3), 12: (0, 4), 13: (1, 4)}
        self.cell2idx = {(1, 2): 7, (0, 0): 0, (3, 3): 11, (3, 0): 2, (3, 1): 5, (2, 1): 4, 
            (0, 2): 6, (1, 3): 9, (2, 3): 10, (1, 4): 13, (2, 2): 8, (0, 4): 12, (1, 0): 1, (1, 1): 3}
    
    def step(self,state,action,slip_chance=True):
        # Input: the current state and action IDs
        # Output: reward, the next state ID, done (episodic terminal boolean value)

        #if np.random.rand() < self.slip:
        #    a = ACTMAP[action]
        #else:
        #    a = action

        a = action
        if slip_chance:
            if np.random.rand() < self.slip:
                a = ACTMAP[action]
        
        cell = self.idx2cell[int(state/8)] 
        if a == 0:
            c_next = cell[1]
            r_next = max(0,cell[0]-1)
        elif a ==1:
            c_next = cell[1]
            r_next = min(self.dim[0]-1,cell[0]+1)
        elif a == 2:
            c_next = max(0,cell[1]-1)
            r_next = cell[0]
        elif a == 3:
            c_next = min(self.dim[1]-1,cell[1]+1)
            r_next = cell[0]
        else:
            print ('action',action,'a:', a) 
            raise ValueError

        if (r_next == self.goal_pos[0]) and (c_next == self.goal_pos[1]): # Reach the exit
            v_flag = self.num2flag(state%8)
            return float(sum(v_flag)), 8*self.cell2idx[(r_next,c_next)] + state%8, True
        else:
            if (r_next,c_next) in Maze.obstacles: # obstacle tuple list
                return 0.0, state, False
            else: # Flag locations
                v_flag = self.num2flag(state%8)
                if (r_next,c_next) == (0,2):
                    v_flag[0] = 1
                elif (r_next,c_next)==(3,0):
                    v_flag[1] = 1
                elif (r_next,c_next) == (3,3):
                    v_flag[2] = 1
                return 0.0, 8*self.cell2idx[(r_next,c_next)] + self.flag2num(v_flag), False


    def num2flag(self,n):
        # n is a positive integer
        # Each element of the below tuple correspond to a status of each flag. 0 for not collected, 1 for collected. 
        flaglist = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
        return list(flaglist[n])

    # It's hard to see this just by looking at this function, but it turns out that
    # this function is the inverse of the num2flag function, which is an easy function to understand
    def flag2num(self,v):
        # v: list
        if sum(v) < 2:
            return np.inner(v,[1,2,3])
        else:
            return np.inner(v,[1,2,3])+1

    def reset(self):
        # Return the initial state
        return 0

    # print out the map
    # and use a different text color for the current location of the robot
    def plot(self,state,action):
        cell = self.idx2cell[int(state/8)] 
        desc = self.map.tolist()

        desc[cell[0]] = desc[cell[0]][:cell[1]] + '\x1b[1;32m' + desc[cell[0]][cell[1]] + '\x1b[0m' + desc[cell[0]][cell[1]+1:]

        #print("action: ", ["UP","DOWN","LEFT","RIGHT"][action] if action is not None else None)
        print("\n".join("".join(row) for row in desc))

### Simple Testing code added by Travers
if __name__ == '__main__':
    maze = Maze()
    # This test shows that maze.flag2num is indeed the inverse of maze.num2flag
    for i in range(8):
        np.testing.assert_almost_equal(i, maze.flag2num(maze.num2flag(i)))
        

### Example main method to allow interaction with the environment
if __name__ == '__main__':
    maze = Maze()

    state = maze.reset()
    maze.plot(state, None)

    while True:
        # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        action = input("Enter Action [0 (for UP), 1 (for DOWN), 2 (for LEFT), 3 (for RIGHT), 4 (to reset), 5 (to quit)]: ")
        action = int(action)
        if action == 5:
            break
        elif action==4:
            state = maze.reset()
            action = None
        else:
            reward, state, done = maze.step(state, action)
            print("Current reward: %s" % reward)
        maze.plot(state, action)
