'''create test set and test model'''
import os
import random
import pickle
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import Environment
from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
torch.set_num_threads(1)

def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)



def test_model(model_range: Union[int, tuple], test_set: Tuple = config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2)

    if isinstance(model_range, int):
        state_dict = torch.load(os.path.join(config.save_path, f'{model_range}.pth'), map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print(f'----------test model {model_range}----------')

        for case in test_set:
            print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
            with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success, steps, num_comm = zip(*ret)


            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("average step: {}".format(sum(steps)/len(steps)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print()

    elif isinstance(model_range, tuple):

        for model_name in range(model_range[0], model_range[1]+1, config.save_interval):
            state_dict = torch.load(os.path.join(config.save_path, f'{model_name}.pth'), map_location=DEVICE)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()


            print(f'----------test model {model_name}----------')

            for case in test_set:
                print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
                with open(f'./test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_one_case, tests)


                success, steps, num_comm = zip(*ret)

                print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
                print("average step: {}".format(sum(steps)/len(steps)))
                print("communication times: {}".format(sum(num_comm)/len(num_comm)))
                print()

            print('\n')
            

def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(np.int)))
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm




def code_test():
    env = Environment()
    network = Network()
    network.eval()
    obs, last_act, pos = env.observe()
    network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(np.int)))

if __name__ == '__main__':

    # load trained model and reproduce results in paper
    test_model(128000)