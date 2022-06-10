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
        observation = env.render()
        return observation


class MinecartWrapper(gym.ObservationWrapper):

    def observation(self, s):
        state = np.append(s['position'], [s['speed'], s['orientation'], *s['content']])
        return state


class DSTModel(nn.Module):

    def __init__(self, nA, scaling_factor, n_hidden=64):
        super(DSTModel, self).__init__()

        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(nn.Linear(110, 64),
                                   nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(3, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        # convert state index to one-hot encoding for Deep Sea Treasure
        state = F.one_hot(state.long(), num_classes=110).to(state.device).float()
        s = self.s_emb(state)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


class WalkroomModel(nn.Module):

    def __init__(self, nS, nA, nO, scaling_factor, n_hidden=64):
        super(WalkroomModel, self).__init__()

        self.nS = nS

        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(nn.Linear(nS, 64),
                                   nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(nO+1, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        state = state.float()
        s = self.s_emb(state)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


class IndexObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super(IndexObservation, self).__init__(env)
        self.sizes = (self.env.size,)*self.env.dimensions

    def observation(self, obs):
        obs = np.ravel_multi_index(obs, self.sizes)
        return obs


class MinecartModel(nn.Module):

    def __init__(self, nA, scaling_factor, n_hidden=64):
        super(MinecartModel, self).__init__()

        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(nn.Linear(6, 64),
                                   nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(4, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


class SumoModel(nn.Module):

    def __init__(self, nA, scaling_factor, n_hidden=64):
        super(SumoModel, self).__init__()

        self.scaling_factor = scaling_factor
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
        self.c_emb = nn.Sequential(nn.Linear(3, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


if __name__ == '__main__':
    import envs
    import torch
    from gym.wrappers import TimeLimit
    import argparse
    from pcn.pcn import train
    from datetime import datetime
    import uuid
    import os
    from main_mones import MultiOneHotEnv

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--env', required=True, type=str, help='dst, minecart, sumo, walkroom2...walkroom9')
    parser.add_argument('--model', default=None, type=str, help='load model')
    args = parser.parse_args()

    device = 'cpu'

    if args.env == 'dst':
        env = gym.make('DeepSeaTreasure-v0')
        env = TimeLimit(env, 200)
        nA = 4
        max_treasure = np.amax(list(env.unwrapped._treasures().values()))
        ref_point = np.array([0, -200.])
        scaling_factor = torch.tensor([[0.1, 0.1, 0.01]]).to(device)
        max_return = np.array([max_treasure, -1.])

        model = DSTModel(nA, scaling_factor).to(device)
        lr, total_steps, batch_size, n_model_updates, n_er_episodes, max_size = 1e-2, 1e5, 256, 10, 50, 200

    elif args.env.startswith('walkroom'):
        nO = int(args.env[len('walkroom'):])
        env = gym.make(f'Walkroom{nO}D-v0')
        env = MultiOneHotEnv(env)
        env = TimeLimit(env, 200)
        nA = nO*2
        ref_point = np.ones(nO)*-env.size
        scaling_factor = torch.tensor([[0.1]*nO+[0.01]]).to(device)
        max_return = np.zeros(nO)

        model = WalkroomModel(env.size*nO, nA, nO, scaling_factor)
        avg_ep_steps = 18 if nO <= 5 else 9
        lr, total_steps, batch_size, n_model_updates, n_er_episodes, max_size = 1e-2, 100*nO*(300+100*nO), 256, 10, 50, 10*nO**3

    elif args.env == 'minecart':
        env = gym.make('MinecartDeterministic-v0')
        env = TimeLimit(env, 1000)
        nA = 6
        ref_point = np.array([0, 0, -200.])
        scaling_factor = torch.tensor([[1, 1, 0.1, 0.1]]).to(device)
        max_return = np.array([1.5, 1.5, -0.])

        model = MinecartModel(nA, scaling_factor).to(device)
        lr, total_steps, batch_size, n_model_updates, n_er_episodes, max_size = 1e-3, 1e7, 256, 50, 20, 50

    elif args.env == 'sumo':
        q_range = 10
        env = gym.make('CrossroadSumo-v0')
        env = TimeLimit(env, max_episode_steps=100)
        env = FrameObservationEnv(env)
        env = CHWEnv(env)
        env = GrayscaleEnv(env)
        env = HistoryEnv(env, size=4)
        env = ScaleRewardEnv(env, min_=np.array([1.2, -0.9]), scale=90/q_range)
        nA = 2
        ref_point = np.array([-2.0, -2.0])*q_range
        scaling_factor = torch.tensor([[1, 1, 0.01]]).to(device)
        max_return = np.array([1.5, 1.5])*q_range

        model = SumoModel(nA, scaling_factor).to(device)
        lr, total_steps, batch_size, n_model_updates, n_er_episodes, max_size = 1e-3, 2e6, 1024, 50, 50, 50

    env.nA = nA

    if args.model is not None:
        model = torch.load(args.model, map_location=device).to(device)
        model.scaling_factor = model.scaling_factor.to(device)

    logdir = f'{os.getenv("LOGDIR", "/tmp")}/pcn/pcn/{args.env}/lr_{lr}/totalsteps_{total_steps}/batch_size_{batch_size}/n_model_updates_{n_model_updates}/n_er_episodes_{n_er_episodes}/max_size_{max_size}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    train(env,
        model,
        learning_rate=lr,
        batch_size=batch_size,
        total_steps=total_steps,
        n_model_updates=n_model_updates,
        n_er_episodes=n_er_episodes,
        max_size=max_size,
        max_return=max_return,
        ref_point=ref_point,
        logdir=logdir)
