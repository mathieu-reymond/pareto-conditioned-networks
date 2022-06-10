from gym.spaces import Box, Discrete, MultiDiscrete
from gym.envs.registration import register
from gym.core import Env
import numpy as np
import itertools
from envs.walkroom.build.grid import initializeGrid


def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    else:
        return solutions[is_efficient]


class WalkRoom(Env):

    def __init__(self, size, dimensions=2, seed=0):
        super(WalkRoom, self).__init__()

        self.size = size
        self.dimensions = dimensions

        self.room = self.make_boundaries(seed=seed)
        # pareto_front_i = [c + (self.room[c],) for c in (np.unravel_index(i, self.room.shape) for i in nd_i)]
        # self.pareto_front_b = np.array(pareto_front_i)*-1

        # possible actions, do not move in diagonal
        self.actions = np.zeros((self.dimensions*2, self.dimensions), dtype=int)
        # move forward
        self.actions[range(self.dimensions), range(self.dimensions)] = 1
        # move backward
        self.actions[range(self.dimensions, self.dimensions*2), range(self.dimensions)] = -1

        self.observation_space = MultiDiscrete(dimensions*(size,))
        self.action_space = Discrete(len(self.actions))
        self.reward_space = Box(-np.inf, 0, (dimensions,))

        self.reset()

    @property
    def pareto_front(self):
        d = self.dimensions-1
        boundary = [coord + (self.room[coord],) for coord in itertools.product(*(np.arange(self.size),)*d)]
        pf = non_dominated(np.array(boundary)*-1)
        return pf

    def make_boundaries(self, seed=0):
        
        d = self.dimensions-1
        room = initializeGrid(self.size, d, seed)
        room = room.reshape((self.size,)*d)
        return room
        
    def reset(self):
        self.pos = np.zeros(self.dimensions, dtype=int)
        return self.pos.copy()

    def step(self, action):
        pos = self.pos + self.actions[action]
        # stay in room bounds
        pos = np.clip(pos, 0, self.size-1)
        reward = -np.abs(self.actions[action].copy())
        # limit of last coord
        limit = self.room[tuple(pos[:-1].tolist())]
        terminal = pos[-1] >= limit
        self.pos = pos
        return pos.copy(), reward, terminal, {}

    def render(self):
        txt = ''
        for y in range(self.size):
            for x in range(self.size):
                if np.all(self.pos == np.array([x, y])):
                    char = 'X'
                else:
                    char = '#' if self.room[x,y] else '.'
                txt += char
            txt += '\n'
        return txt


def random_walkroom(size, jumps=2, seed=0):
    rng = np.random.default_rng(seed)
    # make sure that, on average you have one jump per step
    p = np.ones(jumps)* 1/np.arange(jumps).sum()
    p[0] = 1-(jumps-1)*p[1]
    y = rng.choice(np.arange(jumps), size=size, p=p).cumsum()[::-1]
    return WalkRoom(size, y)


if __name__ == '__main__':
    s, d = 5, 3
    env = WalkRoom(s, d)
    env.reset()
    print(env.pareto_front, len(env.pareto_front))
    # print(env.render())

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    voxels = np.zeros((s,)*d, dtype=bool)
    for coord in itertools.product(*(np.arange(s),)*(d-1)):
        voxels[coord+(env.room[coord],)] = True
    ax.voxels(voxels)
    plt.show()