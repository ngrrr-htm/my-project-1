import numpy as np
import gymnasium as gym
# import matplotlib as plt

class mario (gym.Env):
    def __init__(self, row = 3, column = 5, max_steps = 3*5*2):
        super().__init__()
        self.row = row
        self.column = column
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(4) # what actions are accepted
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(row),gym.spaces.Discrete(column)))
        self.start = (0,0) # starting position
        self.wall = [(1,2), (2,2)] # walls' location
        self.coin = [(0,2), (1,4)] # coins' location
        self.gate = (2,4) # goal

        self.robot_loc = None
        self.coin_sum = None
        self.step_counter = None
        
    def reset(self):
        self.robot_loc = self.start
        self.coin_sum = 0
        self.step_counter = 0
        self.coin = [(0,2), (1,4)]
        return self.robot_loc, {}

    def step (self, action):
        # actions = {0:'up', 1:'down', 2:'left', 3:'right'}
        # I've counted impossible actions as hitting wall as well (for reward reduction)
        row, column = self.robot_loc
        reward = 0
        terminated = False # reaching goal
        truncated = False # reaching max_step
        if action == 0 and row == 0:
            reward -= 49 # plus the -1 reward at the end, which gives 50
        elif action == 0 and row != 0:
            new_row, new_column= row - 1, column
            if (new_row, new_column)  not in self.wall:
                self.robot_loc = (new_row, new_column)
            else:
                reward -= 49
                self.robot_loc = (row, column)

        elif action == 1 and row == self.row - 1:
            reward -= 49
        elif action == 1 and row != self.row - 1:
            new_row, new_column = row + 1, column
            if (new_row, new_column)  not in self.wall:
                self.robot_loc = (new_row, new_column)
            else:
                reward -= 49
                self.robot_loc = (row, column)

        elif action == 2 and column == 0:
            reward -= 49
        elif action == 2 and column != 0:
            new_row, new_column = row, column - 1
            if (new_row, new_column)  not in self.wall:
                self.robot_loc = (new_row, new_column)
            else:
                reward -= 49
                self.robot_loc = (row, column) 

        elif action == 3 and column == self.column - 1:
            reward -= 49
        elif action == 3 and column != self.column - 1:
            new_row, new_column = row, column + 1
            if (new_row, new_column)  not in self.wall:
                self.robot_loc = (new_row, new_column)
            else:
                reward -= 49
                self.robot_loc = (row, column) 
        
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            truncated = True
        reward -= 1
        if self.robot_loc == self.gate:
            reward +=100
            terminated = True
        elif self.robot_loc in self.coin:
            reward += 10
            self.coin.remove(self.robot_loc)
        
        return self.robot_loc, reward, terminated, truncated, {}
        
    def render(self):
        grid = np.full((self.row, self.column), '⬜', dtype=object)

        for x, y in self.wall:
            grid[x, y] = '🧱'

        for x, y in self.coin:
            grid[x, y] = '🪙'

        x, y = self.gate
        grid[x, y] = '🚪'
        
        x, y = self.robot_loc
        grid[x,y] = '🤖'
            
        for row in grid:
            print(' '.join(row))

# env=  mario()
# env.reset()
# # env.step(0)
# env.render()
