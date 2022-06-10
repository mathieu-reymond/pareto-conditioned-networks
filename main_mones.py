import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


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
        state = cv2.resize(state, (42, 42), interpolation=cv2.INTER_AREA)
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


class OneHotEnv(gym.ObservationWrapper):
    
    def __init__(self, env, num_classes=110):
        super(OneHotEnv, self).__init__(env)
        self.num_classes = num_classes

    def observation(self, o):
        oh = np.zeros(self.num_classes)
        oh[o] = 1.
        return oh


class MultiOneHotEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(MultiOneHotEnv, self).__init__(env)
        
    def observation(self, o):
        moh = np.zeros(self.env.size*self.env.dimensions)
        for i, oi in enumerate(o):
            moh[i*self.env.size+oi] = 1.
        return moh


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

        self.s_emb = nn.Sequential(nn.Linear(110, nA),
                                   nn.Sigmoid())

    def forward(self, state):
        s = self.s_emb(state.float())
        return s



class WalkroomModel(nn.Module):

    def __init__(self, nS, nA, n_hidden=64):
        super(WalkroomModel, self).__init__()

        self.s_emb = nn.Sequential(nn.Linear(nS, nA),
                                   nn.Sigmoid(),)

    def forward(self, state):
        s = self.s_emb(state.float())
        return s


class MinecartModel(nn.Module):
    def __init__(self, nA, hidden=64):
        super(MinecartModel, self).__init__()
        self.s_emb = nn.Sequential(nn.Linear(6, 48),
                                   nn.Tanh(),
                                   nn.Dropout(p=0.3),
                                   nn.Linear(48, nA),)

    def forward(self, state):
        state = state.view(len(state), -1)
        state = state/torch.tensor([[1., 1., 1., 360., 1, 1]])

        x = self.s_emb(state.float())
        return x
        

class SumoModel(nn.Module):

    def __init__(self, nA, n_hidden=64):
        super(SumoModel, self).__init__()

        self.s_emb = nn.Sequential(nn.Linear(740*2, 20),
                                   nn.Tanh(),
                                   nn.Linear(20, nA),)

    def forward(self, state):
        state = state - 0.5
        s = self.s_emb(state.float())
        return s


class SumoLanes(gym.ObservationWrapper):

    def observation(self, obs):
        l11 = obs[:, 15:25, :15].flatten()
        l12 = obs[:, 15:25, 25:].flatten()
        l21 = obs[:, :15, 15:25].flatten()
        l22 = obs[:, 25:, 15:25].flatten()
        lc = obs[:, 15:25, 15:25].flatten()
        lanes = np.concatenate((l11, l12, l21, l22, lc))
        return lanes


if __name__ == '__main__':
    import envs
    import torch
    from gym.wrappers import TimeLimit
    import argparse
    from mones.mones import MONES
    from datetime import datetime
    import uuid
    import os
    from ra.wrappers.atari import NormalizedEnv, Rescale42x42
    from ra.wrappers.history import History
    from ra.wrappers.minecart_pixel import PixelMinecart
    from main_pcn import IndexObservation

    parser = argparse.ArgumentParser(description='MONES')
    parser.add_argument('--env', required=True, type=str, help='dst, minecart or sumo')
    parser.add_argument('--model', default=None, type=str, help='load model')
    parser.add_argument('--population', default=None, type=int, help='pop size')
    parser.add_argument('--indicator', default='hypervolume', type=str)
    parser.add_argument('--hidden', default=None, type=int, help='hidden neurons')
    parser.add_argument('--procs', default=1, type=int, help='parallel runs')
    args = parser.parse_args()

    device = 'cpu'

    if args.env == 'dst':
        def make_env():
            env = gym.make('DeepSeaTreasure-v0')
            env = OneHotEnv(env)
            env = TimeLimit(env, 100)
            return env
        nA = 4
        ref_point = np.array([0, -200.])

        model = DSTModel(nA)
        lr, n_population, n_runs, train_iterations, indicator = 1e-1, 50, 1, 500, args.indicator

    elif args.env == 'minecart':
        def make_env():
            env = gym.make('MinecartDeterministic-v0')
            env = TimeLimit(env, 1000)
            return env
        nA = 6
        ref_point = np.array([0, 0, -222.])

        lr, n_population, n_runs, train_iterations, hidden, indicator = 1e-1, 201, 1, 500, 32, args.indicator
        if args.population is not None: n_population = args.population
        if args.hidden is not None: hidden = args.hidden
        model = MinecartModel(nA, hidden).to(device)

    elif args.env.startswith('walkroom'):
        nO = int(args.env[len('walkroom'):])
        def make_env():
            env = gym.make(f'Walkroom{nO}D-v0')
            env = MultiOneHotEnv(env)
            env = TimeLimit(env, 200)
            return env
        env = make_env()
        nA = nO*2
        nS = np.sum(env.observation_space.nvec)
        ref_point = np.ones(nO)*-201 #env.size
        del env

        model = WalkroomModel(nS, nA)
        lr, n_population, n_runs, train_iterations, indicator = 1e-1, 100*nO, 1, 300 + 100*nO, args.indicator
        if args.population is not None: n_population = args.population

    elif args.env == 'sumo':
        q_range = 10
        def make_env():
            env = gym.make('CrossroadSumo-v0')
            env = TimeLimit(env, max_episode_steps=100)
            env = FrameObservationEnv(env)
            env = CHWEnv(env)
            env = GrayscaleEnv(env)
            env = SumoLanes(env)
            env = HistoryEnv(env, size=2)
            env = ScaleRewardEnv(env, min_=np.array([1.2, -0.9]), scale=90/q_range)
            return env
        nA = 2
        ref_point = np.array([-2.0, -2.0])*q_range

        lr, n_population, n_runs, train_iterations, indicator = 1e-1, 50, 1, 1000, args.indicator
        
        if args.population is not None: n_population = args.population
        model = SumoModel(nA).to(device)

    # if args.model is not None:
    #     model = torch.load(args.model).to(device)

    logdir = f'{os.getenv("LOGDIR","/tmp")}/pcn/mones/{args.env}/{args.indicator}/lr_{lr}/population_{n_population}/runs_{n_runs}/train_iterations_{train_iterations}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = MONES(
        make_env,
        model,
        n_population=n_population,
        n_runs=n_runs,
        ref_point=ref_point,
        lr=lr,
        indicator=indicator,
        logdir=logdir,
        n_processes=args.procs,
    )
    agent.train(train_iterations)
