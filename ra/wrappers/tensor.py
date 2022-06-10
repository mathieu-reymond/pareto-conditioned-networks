import gym
import numpy as np
import torch


def TensorWrapper(env):
    return Tensor1DWrapper(env)


class Tensor1DWrapper(gym.Wrapper):

    def __init__(self, env):
        super(Tensor1DWrapper, self).__init__(env)

    def reset(self):
        obs = super(Tensor1DWrapper, self).reset()
        # convert to torch.tensor, needs to be float32 for nn
        obs = torch.from_numpy(obs.astype(np.float32))
        # make 1-sized batch
        obs = obs.unsqueeze(0)
        return obs

    def step(self, action):
        # squeeze to remove 1-sized batch
        action = action.detach().squeeze(0).cpu().numpy()
        # if discrete action, take it from 1-sized array
        if action.dtype != np.float32: action = action[0]
        obs, reward, terminal, info = super(Tensor1DWrapper, self).step(action)
        # convert to torch.tensor, needs to be float32 for nn
        obs = torch.from_numpy(obs.astype(np.float32))
        # add trailing dimension to have [Batch, 1] tensors
        reward = torch.from_numpy(np.array(reward, dtype=np.float32))
        terminal = torch.from_numpy(np.array(terminal)).unsqueeze(-1)
        # make 1-sized batch
        obs, reward, terminal = map(lambda x: x.unsqueeze(0), [obs, reward, terminal])
        return obs, reward, terminal, info