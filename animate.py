from timeit import repeat
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.animation import FuncAnimation

DIR = "logs"
LOG_NAME = "Results" if False else "Neat_Test"
FITNESS_HIDE_FILE = "./logs/fitness/best_hide.npy"
FITNESS_SEEK_FILE = "./logs/fitness/best_seek.npy"

DRAW_SCENT=True
DRAW_REWARDS=True

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

MAP_SIMPLE_NAME = "map.txt"

ACTIONS = np.array([[-1,0], [1, 0], [0,-1], [0, 1]])
ACTIONS_SEEKER =  np.array([[-1,0], [1, 0], [0,-1], [0, 1], [-1,-1], [-1,1], [1,-1],[1,1]])

VISION = 1
SCENT = 5

REWARD =100000

class Map:
        
    def __init__(self,mapname, char_mapping):
        #read map
        with open(mapname, 'r') as map_file:
            lines = map_file.readlines()
            # width = int(lines[0])
            # height = int(lines[1])
            map_numbers = [[char_mapping[chr] for chr in row[:-1]] for row in lines[2:]]
        self.char_mapping =char_mapping
        self.map = np.array(map_numbers, dtype = np.int32)
        self.start_hider_pos = np.asarray(np.where(self.map == char_mapping[HIDER])).reshape(2)
        self.start_seeker_pos = np.asarray(np.where(self.map == char_mapping[SEEKER])).reshape(2)
        
        #action array for indexin
        self.action_indexer = tuple(ACTIONS.T.tolist())
        
        #initialize map and backup map
        self.map_layout = np.array(map_numbers, dtype = np.int32)
        self.map_layout[self.start_hider_pos[0] ,self.start_hider_pos[1] ] = char_mapping[FREE_SPACE]
        self.map_layout[self.start_seeker_pos[0],self.start_seeker_pos[1]] = char_mapping[FREE_SPACE]
        
        #initialize rewards
        hider_reward = np.zeros_like(self.map_layout)-1
        queue = np.where(self.map_layout==char_mapping[SAFE_ZONE])
        hider_reward[queue]=0
        queue = [(x,y)for x,y in zip(*queue)]
        while len(queue)>0:
            new_queue=[]
            for x,y in queue:
                for a_x,a_y in ACTIONS:
                    if (self.get_map_layout_pos([x+a_x,y+a_y])!=char_mapping[WALL] and
                        hider_reward[x+a_x,y+a_y]<0):
                            hider_reward[x+a_x,y+a_y]=hider_reward[x,y]+1
                            new_queue.append((x+a_x,y+a_y))
            queue=new_queue
            
        self.hider_reward =hider_reward/np.max(hider_reward)
          
        self.reset()
    
    def reset(self):
        self.hider_pos  =self.start_hider_pos
        self.seeker_pos =self.start_seeker_pos
        self.map = np.copy(self.map_layout)
        self.set_map_pos(self.hider_pos,  self.char_mapping[HIDER] )
        self.set_map_pos(self.seeker_pos, self.char_mapping[SEEKER])

        self.hider_reward_penalty = np.zeros_like(self.hider_reward)
        self.scent = np.zeros_like(self.map)

    def update(self):
        pos =self.hider_pos
        self.hider_reward_penalty[pos[0], pos[1]]= self.hider_reward[pos[0], pos[1]]
        self.scent[pos[0],pos[1]]=SCENT
        self.scent-=1
        self.scent = np.clip(self.scent, 0, SCENT)
    
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
    
    def get_action_indicies(self, pos):
        return (
            self.action_indexer[0]+pos[0],
            self.action_indexer[1]+pos[1],
        )
    
    def get_percepts(self, agent, vision = 1):
        pos =self.get_agent_pos(agent)
        if agent==HIDER:
            return (self.map[self.get_action_indicies(pos)],[])
        else:
            percepts = self.map[
                pos[0] - vision : pos[0] + vision + 1,
                pos[1] - vision : pos[1] + vision + 1
            ]
            scent= np.append(self.scent[self.get_action_indicies(pos)],self.map[pos[0], pos[1]])
            return (percepts,scent) 

    def is_free(self, pos, agent):
        map_pos =self.get_map_layout_pos(pos)
        if (agent == HIDER):
            return (
                map_pos in [self.char_mapping[FREE_SPACE], self.char_mapping[SAFE_ZONE]] and
                not np.all(pos == self.seeker_pos)
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
        usable_actions = ACTIONS_SEEKER if agent == SEEKER else ACTIONS
        new_pos = pos + usable_actions[action]
        if self.is_free(new_pos, agent):
            self.move_agent(agent, new_pos)
    
    def is_end(self):
        return np.all(self.hider_pos == self.seeker_pos)

    def get_reward(self,pos):
         return max(self.hider_reward[pos[0], pos[1]] - self.hider_reward_penalty[pos[0], pos[1]],0)

map = Map(MAP_SIMPLE_NAME,CHAR_MAPPING)

path =os.path.join(DIR, LOG_NAME)
games = os.listdir(path)

if(len(games)%2==0):
    games =np.concatenate(
        np.take_along_axis(
            np.array(games).reshape(2,-1),
            np.argsort(np.array([
                int("".join(i for i in game if i.isdigit())) 
                for game in games
                ]).reshape(2,-1), axis=1,kind="stable"),
            axis=1
        ).T)

for game in games:
    moves = np.load(os.path.join(path, game))
    fig, ax = plt.subplots()
    ax.set_title("Generation "+game)
    im = plt.imshow(map.map, cmap = matplotlib.colors.ListedColormap(COLOR_MAPPING))

    ssc =plt.scatter([],[],c=[],s=400,marker="o",cmap="YlGn",vmin=0, vmax=SCENT)
    rsc =plt.scatter([],[],c=[],s=100,marker="o",cmap="Wistia",vmin=0, vmax=np.max(map.hider_reward))
    
    def init():
        map.reset()
        if DRAW_REWARDS:
            rewards = map.hider_reward-map.hider_reward_penalty
            y,x =np.where(rewards>0)
            rsc.set_offsets(np.vstack((x,y)).T)
            rsc.set_array(rewards[(y,x)])
        return [im,ssc,rsc]

    def update(move):
        map.update()
        map.move_agent(HIDER,move[0])
        map.move_agent(SEEKER,move[1])
        im.set_data(map.map)
        if DRAW_SCENT:
            y,x =np.where(map.scent>0)
            ssc.set_offsets(np.vstack((x,y)).T)
            ssc.set_array(map.scent[(y,x)])
        if DRAW_REWARDS:
            rewards = map.hider_reward-map.hider_reward_penalty
            y,x =np.where(rewards>0)
            rsc.set_offsets(np.vstack((x,y)).T)
            rsc.set_array(rewards[(y,x)])
        return [im,ssc,rsc]

    ani = FuncAnimation(fig, update, frames=moves,
                        init_func=init, blit=True, repeat=False)
    plt.show()
fits_seek = np.load(FITNESS_SEEK_FILE)
fits_hide = np.load(FITNESS_HIDE_FILE)
plt.plot(np.arange(0, fits_seek.shape[0]), fits_seek)
plt.show()
plt.plot(np.arange(0, fits_hide.shape[0]), fits_hide)
plt.show()
