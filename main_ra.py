import gym
import envs.walkroom
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from gym.wrappers import TimeLimit
import numpy as np
from main_mones import OneHotEnv, MultiOneHotEnv
from main_pcn import IndexObservation


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_=0., scale=1.):
        gym.RewardWrapper.__init__(self, env)
        self.min = min_
        self.scale = scale

    def reward(self, reward):
        return (reward - self.min)/self.scale


class CHWEnv(gym.ObservationWrapper):

    def observation(self, observation):
        # from whc to chw
        return np.moveaxis(observation, [1, 0, 2], [2, 1, 0])


class GrayscaleEnv(gym.ObservationWrapper):
    """
    Expects a state-image, in CHW, with 3 channels: in RGB
    If the state is in WHC, use the CHWEnv wrapper first
    """

    def observation(self, state):
        # RGB to grayscale
        r, g, b = state[0], state[1], state[2]
        state = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # rescale to (84, 84)
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
        # normalize state
        state /= 255.
        # add channel dim
        state = np.expand_dims(state, 0)

        return state


class HistoryEnv(gym.Wrapper):
    def __init__(self, env, size=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.size = size
        # will be set in _convert
        self._state = None

        # history stacks observations on dim 0
        low = np.repeat(self.observation_space.low, self.size, axis=0)
        high = np.repeat(self.observation_space.high, self.size, axis=0)
        self.observation_space = gym.spaces.Box(low, high)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        state = self.env.reset(**kwargs)
        # add history dimension
        s = np.expand_dims(state, 0)
        # fill history with current state
        self._state = np.repeat(s, self.size, axis=0)
        return np.concatenate(self._state, axis=0)

    def step(self, ac):
        state, r, d, i = self.env.step(ac)
        # shift history
        self._state = np.roll(self._state, -1, axis=0)
        # add state to history
        self._state[-1] = state
        return np.concatenate(self._state, axis=0), r, d, i


class FrameObservationEnv(gym.ObservationWrapper):

    def observation(self, observation):
        # ignore observation, render frame and use that instead
        observation = self.env.render()
        return observation


class MinecartWrapper(gym.ObservationWrapper):

    def observation(self, s):
        state = np.append(s['position'], [s['speed'], s['orientation'], *s['content']])
        return state


class DSTModel(nn.Module):

    def __init__(self, nA, n_hidden=64):
        super(DSTModel, self).__init__()

        self.s_emb = nn.Sequential(nn.Linear(110, 20),
                                   nn.Tanh(),
                                   nn.Linear(20, nA),
                                   nn.LogSoftmax(-1))

    def forward(self, state):
        s = self.s_emb(state)
        return s


class WalkroomModel(nn.Module):

    def __init__(self, nS, nA, n_hidden=64):
        super(WalkroomModel, self).__init__()

        self.s_emb = nn.Sequential(nn.Linear(nS, 20),
                                   nn.Tanh(),
                                   nn.Linear(20, nA),
                                   nn.LogSoftmax(-1))

    def forward(self, state):
        s = self.s_emb(state)
        return s


class MinecartModel(nn.Module):

    def __init__(self, nA, hidden=64):
        super(MinecartModel, self).__init__()

        self.s_emb = nn.Sequential(nn.Linear(6, hidden),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden),
                                   nn.Tanh(),
                                   nn.Linear(hidden, nA),
                                   nn.LogSoftmax(1))

    def forward(self, state):
        x = self.s_emb(state.float())
        return x


class SumoModel(nn.Module):

    def __init__(self, nA, n_hidden=64):
        super(SumoModel, self).__init__()

        self.s_emb = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 64),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state):
        s = self.s_emb(state.float())
        s = self.fc(s)
        return s


def make_dst_env():
    env = gym.make('DeepSeaTreasure-v0')
    env = OneHotEnv(env, num_classes=110)
    env = TimeLimit(env, 100)
    return env


if __name__ == '__main__':
    import envs
    import numpy as np
    import torch
    import argparse
    from ra.ra import RA
    from ra.memory import EpisodeMemory
    from ra.policy import Categorical
    from datetime import datetime
    import uuid
    import os

    parser = argparse.ArgumentParser(description='RA')
    parser.add_argument('--env', required=True, type=str, help='dst, minecart or sumo')
    parser.add_argument('--model', default=None, type=str, help='load model')
    parser.add_argument('--population', default=None, type=int, help='pop size')
    parser.add_argument('--timesteps', default=None, type=int, help='timesteps for each pareto ascent training')
    parser.add_argument('--hidden', default=None, type=int, help='hidden neurons')
    args = parser.parse_args()

    device = 'cpu'

    if args.env == 'dst':
        make_env = make_dst_env
        nA = 4

        model = DSTModel(nA)
        lr, n_population, timesteps = 1e-3, 10, 100000
        if args.population is not None: n_population = args.population
        if args.timesteps is not None: timesteps = args.timesteps
    
    elif args.env.startswith('walkroom'):
        nO = int(args.env[len('walkroom'):])
        def make_env():
            env = gym.make(f'Walkroom{nO}D-v0')
            env = MultiOneHotEnv(env)
            env = TimeLimit(env, 200)
            return env
        nA = nO*2
        nS = make_env().size*nO # np.prod(make_env().observation_space.nvec)

        
        model = WalkroomModel(nS, nA).to(device)
        lr, n_population, timesteps = 1e-3, 32, 100000
        if args.population is not None: n_population = args.population
        if args.timesteps is not None: timesteps = args.timesteps

    elif args.env == 'minecart':
        def make_env():
            env = gym.make('MinecartDeterministic-v0')
            # env = MinecartOneHot(env)
            env = TimeLimit(env, 1000)
            return env
        nA = 6

        lr, n_population, timesteps, hidden = 3e-4, 10, 20000000, 20
        if args.population is not None: n_population = args.population
        if args.timesteps is not None: timesteps = args.timesteps
        model = MinecartModel(nA, hidden).to(device)

    elif args.env == 'sumo':
        q_range = 10
        def make_env():
            env = gym.make('CrossroadSumo-v0')
            env = TimeLimit(env, max_episode_steps=100)
            env = FrameObservationEnv(env)
            env = CHWEnv(env)
            env = GrayscaleEnv(env)
            env = HistoryEnv(env, size=4)
            env = ScaleRewardEnv(env, min_=np.array([1.2, -0.9]), scale=90/q_range)
            return env
        nA = 2
        ref_point = np.array([-2.0, -2.0])*q_range

        model = SumoModel(nA).to(device)
        lr, n_population, timesteps, hidden = 3e-4, 50, 2000000, 64
        if args.population is not None: n_population = args.population
        if args.timesteps is not None: timesteps = args.timesteps

    logdir = f'{os.getenv("LOGDIR", "/tmp")}/pcn/ra/{args.env}/lr_{lr}/population_{n_population}/timesteps_{timesteps}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = RA(
        make_env,
        actor=model,
        policy=Categorical(),
        memory=EpisodeMemory(),
        n_processes=n_population,
        gamma=1.,
        lr=lr,
        logdir=logdir,
        clip_grad_norm=50,
    )
    print(agent.lambdas)
    agent.train(timesteps=timesteps)
