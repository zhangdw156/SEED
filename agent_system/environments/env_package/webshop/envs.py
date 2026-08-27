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

import ray
import gym
import numpy as np
from typing import Any

from agent_system.environments.fairness import (
    webshop_goal_fingerprint,
    webshop_goal_indices,
    webshop_reset_goal_indices,
)

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs = dict(env_kwargs)
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
        goals = getattr(getattr(self.env, 'server', None), 'goals', None)
        self._goal_fingerprint = (
            None if goals is None else webshop_goal_fingerprint(goals)
        )
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals

    def get_goal_fingerprint(self):
        if self._goal_fingerprint is None:
            raise ValueError(
                'WebShop environment does not expose resolved goals'
            )
        return self._goal_fingerprint
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        rng: np.random.RandomState | None = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = rng if rng is not None else np.random.RandomState(seed)

        self._env_kwargs: dict[str, Any] = dict(
            env_kwargs
            if env_kwargs is not None
            else {'observation_mode': 'text', 'num_products': None}
        )
        configured_goal_indices = self._env_kwargs.pop(
            'fairness_goal_indices',
            None,
        )
        configured_fairness_split = self._env_kwargs.pop(
            'fairness_split',
            None,
        )
        self.fairness_enabled = bool(
            self._env_kwargs.pop('fairness', True)
        )
        if self.fairness_enabled:
            self._env_kwargs['goal_seed'] = 0

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        if self.fairness_enabled:
            fingerprints = ray.get([
                worker.get_goal_fingerprint.remote()
                for worker in self._workers
            ])
            if len(set(fingerprints)) != 1:
                for worker in self._workers:
                    ray.kill(worker)
                raise ValueError(
                    'WebShop workers resolved different canonical goal lists'
                )

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if self.fairness_enabled:
            fairness_split = (
                configured_fairness_split
                or ('train' if self.is_train else 'evaluation')
            )
            self.goal_idxs: list[int] = (
                [int(value) for value in configured_goal_indices]
                if configured_goal_indices is not None
                else webshop_goal_indices(fairness_split)
            )
            if max(self.goal_idxs) >= len(goals):
                raise ValueError(
                    f'WebShop fairness goal index '
                    f'{max(self.goal_idxs)} exceeds '
                    f'the canonical goal count {len(goals)}'
                )
        elif not self.is_train:
            self.goal_idxs = list(range(500))
        else:
            self.goal_idxs = list(range(500, len(goals)))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        return self.step_selected(actions, list(range(self.num_processes)))

    def step_selected(self, actions: list[str], indices: list[int]):
        if len(actions) != len(indices):
            raise ValueError(
                f'Expected one action per selected environment, got '
                f'{len(actions)} actions for {len(indices)} environments',
            )
        if any(index < 0 or index >= self.num_processes for index in indices):
            raise ValueError('Selected environment index is out of range')

        futures = [
            self._workers[index].step.remote(action)
            for action, index in zip(actions, indices)
        ]

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = webshop_reset_goal_indices(
            is_train=self.is_train,
            fairness_enabled=self.fairness_enabled,
            goal_indices=self.goal_idxs,
            env_num=self.env_num,
            group_n=self.group_n,
            rng=self._rng,
        )

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    rng: np.random.RandomState | None = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        rng=rng,
    )
