from gym.envs.toy_text import discrete
from gym.spaces.box import Box
import numpy as np
import cv2


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class DeepSeaTreasureEnv(discrete.DiscreteEnv):
    """ A classic multi-objective environment.
        A submarine search for treasures hidden in the sea, at the expense of fuel consumption.
        
        Code from:
        https://gitlab.ai.vub.ac.be/mreymond/deep-sea-treasure
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=10):

        self.shape = (width+1, width)
        self.start_state_index = 0

        nS = np.prod(self.shape)
        nA = 4

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        max_treasure = np.amax(list(self._treasures().values()))
        self.reward_space = Box(low=np.array([0., -1.]), high=np.array([max_treasure, -1.]), dtype=np.float32)

        super(DeepSeaTreasureEnv, self).__init__(nS, nA, P, isd)

    def _treasures(self):

        if self.shape[1] > 10:
            raise ValueError('Default Deep Sea Treasure only supports a grid-size of max 10')

        return {(1, 0): 1,
                (2, 1): 2,
                (3, 2): 3,
                (4, 3): 5,
                (4, 4): 8,
                (4, 5): 16,
                (7, 6): 24,
                (7, 7): 50,
                (9, 8): 74,
                (10, 9): 124}

    def _unreachable_positions(self):
        u = []
        treasures = self._treasures()
        for p in treasures.keys():
            for i in range(p[0]+1, self.shape[0]):
                u.append((i, p[1]))
        return u

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):

        unreachable = self._unreachable_positions()
        treasures = self._treasures()
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_position = tuple(new_position)
        if new_position in unreachable:
            new_position = tuple(current)
        new_state = np.ravel_multi_index(new_position, self.shape)

        if new_position in treasures:
            reward = [treasures[new_position], -1]
            done = True
        else:
            reward = [0, -1]
            done = False
        return [(1., new_state, np.array(reward, dtype=np.float32), done)]

    def render(self, mode='rgb_array'):
        tile_size = 20; font_size = tile_size/75
        img = np.full((self.shape[0]*tile_size, self.shape[1]*tile_size, 3), 255, np.uint8)
        # get ground contour
        coords = np.array([k for k in self._treasures().keys()])
        # compute end of each treasure coordinate to make contour,
        # swap y-x values, as y is first in treasure dict
        coords = coords[:, [1, 0]].repeat(2, axis=0)
        coords[1:-1:2, 0] = coords[2::2, 0]
        coords = coords*tile_size + np.array([-1, 0])
        x_lim, y_lim = img.shape[1], img.shape[0]
        coords[-1, 0] = x_lim
        sea_coords = np.concatenate((coords, np.array([[x_lim, 0], [0, 0]])))
        treasure_coords = np.concatenate((coords, coords[::-1] + np.array([0, tile_size])))
        bottom_coords = np.array([[x_lim-1, y_lim], [0, y_lim]])
        bottom_coords = np.concatenate((coords+ np.array([0, tile_size]), bottom_coords))
        cv2.fillPoly(img, sea_coords.astype(np.int32)[None], (255, 0, 0))
        cv2.fillPoly(img, treasure_coords.astype(np.int32)[None], (0, 255, 0))
        cv2.fillPoly(img, bottom_coords.astype(np.int32)[None], (0, 0, 255))
        # put treasure values
        for c, t in self._treasures().items():
            textsize = cv2.getTextSize(str(t), cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)[0]
            treasure_coord = (tile_size*c[1]+tile_size//2-textsize[0]//2, tile_size*c[0]+tile_size//2+textsize[1]//2)
            cv2.putText(img, str(t), treasure_coord, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0))
        # put submarine
        p = np.unravel_index(self.s, self.shape)
        p = [pi*tile_size for pi in p]
        cv2.rectangle(img, (p[1], p[0]), (p[1]+tile_size, p[0]+tile_size), (255, 255, 255), -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class BountyfulSeaTreasureEnv(DeepSeaTreasureEnv):

    def __init__(self, width=10):
        # ensure that same depths will always be chosen
        # random treasure-depths for each x-pos
        depths = np.random.RandomState(0).choice(range(4), size=width-1, p=[.3, .5, .1, .1])
        # add first treasure depth (always 1)
        depths = np.append([1], depths)
        depths = np.cumsum(depths)
        # limit to grid
        depths[depths > width] = width
        self.depths = depths
        super(BountyfulSeaTreasureEnv, self).__init__(width=width)

    def _treasures(self):

        pareto_front = lambda x: np.round(-45.64496 - (59.99308/-0.2756738)*(1 - np.exp(0.2756738*x)))

        return {(d, i): pareto_front(-(i+d)) for i, d in enumerate(self.depths)}


class ConvexSeaTreasureEnv(DeepSeaTreasureEnv):

    def __init__(self, width=11):
        super(ConvexSeaTreasureEnv, self).__init__(width=width)

    def _treasures(self):
        return {(2, 0): 18,
                (2, 1): 26,
                (2, 2): 31,
                (4, 3): 44,
                (4, 4): 48.2,
                (5, 5): 56,
                (8, 6): 72,
                (8, 7): 76.3,
                (10, 8): 90,
                (11, 9): 100}