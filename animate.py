from timeit import repeat
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.animation import FuncAnimation

DIR = "logs"
LOG_NAME = "Neat_Test"
MAP = "map.txt"

WALL      = "x"
FREE_SPACE= "."
SAFE_ZONE = "H"
HIDER     = "h" 
SEEKER    = "s"

CHAR_MAPPING = {
    WALL      : -1,
    FREE_SPACE:  0,
    SAFE_ZONE :  1,
    HIDER     :  2, 
    SEEKER    :  3
}

COLOR_MAPPING = ["black", "white", "blue", "green", "red"]

ACTIONS = [[0,0], [-1,0], [1, 0], [0,-1], [0, 1]]
VISION = 1
SCENT = 5

class Map:
        
    def __init__(self,mapname, char_mapping):
        with open(mapname, 'r') as map_file:
            lines = map_file.readlines()
            # width = int(lines[0])
            # height = int(lines[1])
            map_numbers = [[char_mapping[chr] for chr in row[:-1]] for row in lines[2:]]
        self.char_mapping =char_mapping
        self.map = np.array(map_numbers, dtype = np.int32)
        self.scent = np.zeros_like(self.map)
        self.start_hider_pos = np.asarray(np.where(self.map == char_mapping[HIDER])).reshape(2)
        self.start_seeker_pos = np.asarray(np.where(self.map == char_mapping[SEEKER])).reshape(2)
        
        self.map_layout = np.array(map_numbers, dtype = np.int32)
        self.map_layout[self.start_hider_pos[0] ,self.start_hider_pos[1] ] = char_mapping[FREE_SPACE]
        self.map_layout[self.start_seeker_pos[0],self.start_seeker_pos[1]] = char_mapping[FREE_SPACE]
        
        self.reset()
    
    def reset(self):
        self.hider_pos  =self.start_hider_pos
        self.seeker_pos =self.start_seeker_pos
        self.map = np.copy(self.map_layout)
        self.set_map_pos(self.hider_pos,  self.char_mapping[HIDER] )
        self.set_map_pos(self.seeker_pos, self.char_mapping[SEEKER])

    def get_map_layout_pos(self,pos):
        return self.map_layout[pos[0], pos[1]]
    
    def set_map_pos(self,pos, value):
        self.map[pos[0], pos[1]] = value
    
    def get_agent_pos(self, agent):
        return self.hider_pos if agent ==HIDER else self.seeker_pos
    
    def set_agent_pos(self, agent, pos):
        if agent ==HIDER:
            self.hider_pos  =pos
        else:
            self.seeker_pos =pos
    
    def get_percepts(self, agent,vision = 1):
        vision = VISION if agent ==HIDER else VISION
        pos =self.get_agent_pos(self, agent)
        percepts = self.map[
            pos[0] - vision : pos[0] + vision + 1,
            pos[1] - vision : pos[1] + vision + 1
        ]
        return percepts 

    def is_free(self, pos, agent):
        map_pos =self.get_map_layout_pos(pos)
        if (agent == HIDER):
            return (
                map_pos in [self.char_mapping[FREE_SPACE], self.char_mapping[SAFE_ZONE]] and
                np.all(pos != self.seeker_pos)
            )
        else:
            return map_pos in [self.char_mapping[FREE_SPACE]]
            
    def move_agent(self, agent, new_pos):
        pos = self.get_agent_pos(agent)
        self.set_map_pos(pos,self.get_map_layout_pos(pos))
        self.set_map_pos(new_pos,self.char_mapping[agent])
        self.set_agent_pos( agent, new_pos)

    def do_action(self, agent, action):
        pos = self.get_agent_pos(agent)
        new_pos = pos + ACTIONS[action]
        if self.is_free(new_pos, agent):
            self.move_agent(agent, new_pos)
    
    def is_end(self):
        return np.all(self.hider_pos == self.seeker_pos)

map = Map(MAP,CHAR_MAPPING)
path =os.path.join(DIR, LOG_NAME)
games = os.listdir(path)
for game in games:
    print(game)
    moves = np.load(os.path.join(path, game))
    fig, ax = plt.subplots()
    ax.set_title("Generation "+game)
    im = plt.imshow(map.map, cmap = matplotlib.colors.ListedColormap(COLOR_MAPPING))

    def init():
        map.reset()
        return [im]

    def update(move):
        map.move_agent(HIDER,move[0])
        map.move_agent(SEEKER,move[1])
        im.set_data(map.map)
        return [im]

    ani = FuncAnimation(fig, update, frames=moves,
                        init_func=init, blit=True, repeat=False)
    plt.show()

