# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
from agent_system.environments.fairness import (
    alfworld_fairness_split,
    alfworld_gamefiles,
    alfworld_worker_gamefiles,
)

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, game_file=None):
        if game_file is not None:
            base_env.game_files = [game_file]
            base_env.num_games = 1
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
        super().__init__()
        env_kwargs = {} if env_kwargs is None else env_kwargs
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        fairness_enabled = bool(env_kwargs.get('fairness', True))
        canonical_game_files = None
        if fairness_enabled:
            fairness_split = alfworld_fairness_split(
                is_train=is_train,
                eval_dataset=eval_dataset,
                requested_split=env_kwargs.get('fairness_split'),
            )
            configured_game_files = env_kwargs.get('fairness_game_files')
            canonical_game_files = [
                os.path.expanduser(os.path.expandvars(path))
                for path in (
                    configured_game_files
                    if configured_game_files is not None
                    else alfworld_gamefiles(fairness_split)
                )
            ]
            config['dataset']['game_files'] = canonical_game_files
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        if canonical_game_files is not None:
            setattr(base_env, "game_files", canonical_game_files)
            setattr(base_env, "num_games", len(canonical_game_files))
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        worker_game_files = (
            [None] * self.num_processes
            if canonical_game_files is None
            else alfworld_worker_gamefiles(
                fixed_assignment=not is_train,
                canonical_gamefiles=canonical_game_files,
                num_processes=self.num_processes,
            )
        )

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i, game_file in enumerate(worker_game_files):
            worker = env_worker.remote(
                config,
                seed + (i // self.group_n),
                base_env,
                game_file,
            )
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"
        return self.step_selected(actions, list(range(self.num_processes)))

    def step_selected(self, actions, indices):
        assert len(actions) == len(indices), \
            "The num of actions must be equal to the num of selected processes"
        assert all(0 <= index < self.num_processes for index in indices), \
            "Selected process index is out of range"

        futures = [
            self.workers[index].step.remote(action)
            for action, index in zip(actions, indices)
        ]

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for index, (obs, scores, dones, info) in zip(indices, results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[index] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs_selected(indices)
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    def getobs_selected(self, indices):
        futures = [self.workers[index].getobs.remote() for index in indices]
        return ray.get(futures)

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
