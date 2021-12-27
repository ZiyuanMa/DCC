import random
from typing import List, Union
import numpy as np
import config

ACTION_LIST = np.array([[-1, 0],[1, 0],[0, -1],[0, 1], [0, 0]], dtype=np.int)
    
class Environment:
    def __init__(self, num_agents: int = config.init_env_settings[0], map_length: int = config.init_env_settings[1],
                obs_radius: int = config.obs_radius, reward_fn: dict = config.reward_fn, fix_density=None,
                curriculum=False, init_env_settings_set=config.init_env_settings):

        self.curriculum = curriculum
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.map_size = (init_env_settings_set[1], init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        # set as same as in PRIMAL
        if fix_density is None:
            self.fix_density = False
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        else:
            self.fix_density = True
            self.obstacle_density = fix_density

        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)
        
        partition_list = self._map_partition()

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)
            partition_list = self._map_partition()
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([len(partition) for partition in partition_list])
        
        # loop to assign agent original position and goal position for each agent
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn
        self._get_heuri_map()
        self.steps = 0

        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.bool)

    
    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, map_length=None):

        if self.curriculum:
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            self.map_size = (rand[1], rand[1])

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        if not self.fix_density:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        
        partition_list = self._map_partition()

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = self._map_partition()
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([len(partition) for partition in partition_list])
        
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self._get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.bool)

        return self.observe()

    def load(self, map:np.ndarray, agents_pos:np.ndarray, goals_pos:np.ndarray):

        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])
        
        self.steps = 0

        self._get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.bool)

    def _get_heuri_map(self):
        dist_map = np.ones((self.num_agents, *self.map_size), dtype=np.int32) * np.iinfo(np.int32).max

        empty_pos = np.argwhere(self.map==0).tolist()
        empty_pos = set([tuple(pos) for pos in empty_pos])

        for i in range(self.num_agents):
            open_list = set()
            x, y = tuple(self.goals_pos[i])
            open_list.add((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop()
                dist = dist_map[i, x, y]

                up = x-1, y
                if up in empty_pos and dist_map[i, x-1, y] > dist+1:
                    dist_map[i, x-1, y] = dist+1
                    open_list.add(up)
                
                down = x+1, y
                if down in empty_pos and dist_map[i, x+1, y] > dist+1:
                    dist_map[i, x+1, y] = dist+1
                    open_list.add(down)
                
                left = x, y-1
                if left in empty_pos and dist_map[i, x, y-1] > dist+1:
                    dist_map[i, x, y-1] = dist+1
                    open_list.add(left)
                
                right = x, y+1
                if right in empty_pos and dist_map[i, x, y+1] > dist+1:
                    dist_map[i, x, y+1] = dist+1
                    open_list.add(right)
        
        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=np.bool)

        for x, y in empty_pos:
            for i in range(self.num_agents):

                if x > 0 and dist_map[i, x-1, y] < dist_map[i, x, y]:
                    self.heuri_map[i, 0, x, y] = 1
                
                if x < self.map_size[0]-1 and dist_map[i, x+1, y] < dist_map[i, x, y]:
                    self.heuri_map[i, 1, x, y] = 1

                if y > 0 and dist_map[i, x, y-1] < dist_map[i, x, y]:
                    self.heuri_map[i, 2, x, y] = 1
                
                if y < self.map_size[1]-1 and dist_map[i, x, y+1] < dist_map[i, x, y]:
                    self.heuri_map[i, 3, x, y] = 1

        self.heuri_map = np.pad(self.heuri_map, ((0, 0), (0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)))
    
    def _map_partition(self):
        '''
        partitioning map into independent partitions 
        '''
        empty_list = np.argwhere(self.map==0).tolist()

        empty_pos = set([tuple(pos) for pos in empty_list])

        if not empty_pos:
            raise RuntimeError('no empty position')

        partition_list = list()
        while empty_pos:

            start_pos = empty_pos.pop()

            open_list = list()
            open_list.append(start_pos)
            close_list = list()

            while open_list:
                x, y = open_list.pop(0)

                up = x-1, y
                if up in empty_pos:
                    empty_pos.remove(up)
                    open_list.append(up)
                
                down = x+1, y
                if down in empty_pos:
                    empty_pos.remove(down)
                    open_list.append(down)
                
                left = x, y-1
                if left in empty_pos:
                    empty_pos.remove(left)
                    open_list.append(left)
                
                right = x, y+1
                if right in empty_pos:
                    empty_pos.remove(right)
                    open_list.append(right)

                close_list.append((x, y))

            if len(close_list) >= 2:
                partition_list.append(close_list)

        return partition_list

    def step(self, actions: List[int]):
        '''
        actions:
            list of indices
                0 up
                1 down
                2 left
                3 right
                4 stay
        '''

        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(len(actions), self.num_agents)
        assert all([action_idx<5 and action_idx>=0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 4:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += ACTION_LIST[actions[agent_id]]
                rewards.append(self.reward_fn['move'])

        # first round check, these two conflicts have the highest priority
        for agent_id in checking_list.copy():

            if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))[0]

                if target_agent_id:

                    target_agent_id = target_agent_id.item()

                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:

                collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0]*self.map_size[0]+x[1])

                        collide_agent_id.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]
        else:
            done = False

        info = {'step': self.steps-1}

        # make sure no overlapping agents
        assert np.unique(self.agents_pos, axis=0).shape[0] == self.num_agents

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5), dtype=np.bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info


    def observe(self):
        '''
        return observation and position for each agent

        obs: shape (num_agents, 6, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map 
            layer 2: obstacle map
            layer 3-6: heuristic map
        
        last_act: agents' last step action

        pos: current position of each agent, used for caculating communication mask

        '''
        obs = np.zeros((self.num_agents, 6, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)

        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)

        agent_map = np.zeros((self.map_size), dtype=np.bool)
        agent_map[self.agents_pos[:,0], self.agents_pos[:,1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
            obs[i, 2:] = self.heuri_map[i, :, x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]

        return obs, np.copy(self.last_actions), np.copy(self.agents_pos)