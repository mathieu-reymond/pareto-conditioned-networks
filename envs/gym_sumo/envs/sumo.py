import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('please declare environment variable \'SUMO_HOME\'')

import libsumo
import gym
from scipy.ndimage.interpolation import rotate
import numpy as np
import cv2


class SumoEnv(gym.Env):

    def __init__(self, sumocfg):
        super(SumoEnv, self).__init__()
        self.sumocfg = sumocfg

        # need to start sumo to get network information
        libsumo.start(['-c', self.sumocfg])
        self.light_ids = libsumo.trafficlight.getIDList()
        self.lights_actions = self._lights_actions()
        self.vehicle_ids = {}

        self.observation_space = gym.spaces.Box(low=-4, high=np.inf, shape=self._grid().shape)
        self.action_space = gym.spaces.Discrete(2)
        libsumo.close()

    def _lights_actions(self):
        """
        temp hack as libsumo has no interface for trafficlight programs yet.
        Match each phase without yellow light with an action
        """
        # increase phase number until exception to get number of phases
        light_actions = []
        for l in self.light_ids:
            phase = 0
            actions = []
            while True:
                try:
                    libsumo.trafficlight.setPhase(l, phase)
                    ryg = libsumo.trafficlight.getRedYellowGreenState(l)
                    # only add phase to action if it's not a yellow light
                    if not 'y' in ryg:
                        actions.append(phase)
                    phase += 1
                # phase does not exist, tried all phases
                except libsumo.libsumo.TraCIException:
                    break
            light_actions.append(actions)
        return light_actions

    def _vehicles_delta(self):
        v_id = {id_: 1 for id_ in libsumo.vehicle.getIDList()}
        v_common = self.vehicle_ids.keys() & v_id.keys()
        reached_destination = 0
        # for each vehicle that was already there, check if they reached destination
        for v in v_common:
            road = libsumo.vehicle.getRoadID(v)
            target = libsumo.vehicle.getRoute(v)[-1]
            # cars on target lane are still present in simulator, but should be ignored in grid
            if self.vehicle_ids[v] > 0:
                # if reached destination, set waiting-time to zero
                if road == target:
                    v_id[v] = 0
                    reached_destination += 1
                else:
                    v_id[v] += self.vehicle_ids[v]
            else:
                v_id[v] = 0
        # updated all car's waiting times
        self.vehicle_ids = v_id
        return reached_destination

    def _grid(self):
        SCALE = 0.25
        bounds = np.array(libsumo.simulation.getNetBoundary())
        size = np.abs(np.diff(bounds, axis=0)*SCALE).astype(np.int).flatten()
        grid = np.zeros(size)
        # get the properties of each vehicle and add them to the grid
        for v, w in self.vehicle_ids.items():
            # pos relative to 0
            pos = (np.array(libsumo.vehicle.getPosition(v)) - bounds[0])*SCALE
            x, y = pos.astype(np.int32)
            grid[x, y] = w
            # angle in degrees
            # deg = libsumo.vehicle.getAngle(v)
            # # TODO vehicle as a single point or as small triangle/rectangle?
            # # vehicle upwards
            # vehicle = np.array([[0,0,0],[1,1,1],[0,0,0]])
            # vehicle = rotate(vehicle, deg)[:3,:3]
            # # put vehicle on grid
            # gs = grid[x-1:x+2,y-1:y+2].shape
            # grid[x-1:x+2,y-1:y+2] = vehicle[:gs[0],:gs[1]]*w

        for tl in self.light_ids:
            # assume the light controls one junction
            j = libsumo.trafficlight.getControlledJunctions(tl)[0]
            j_shape = (np.array(libsumo.junction.getShape(j)) - bounds[0])*SCALE
            xmin, xmax = np.amin(j_shape[:,0]), np.amax(j_shape[:,0])
            ymin, ymax = np.amin(j_shape[:,1]), np.amax(j_shape[:,1])
            xmin, xmax = int(xmin), int(xmax)
            ymin, ymax = int(ymin), int(ymax)
            # lights are represented by their phase
            # negative value to distinguish with vehicle waiting time
            p = (libsumo.trafficlight.getPhase(tl) + 1)*-1
            grid[xmin, ymin] = p
            grid[xmin, ymax] = p
            grid[xmax, ymin] = p
            grid[xmax, ymax] = p

        return np.expand_dims(grid.T, 0)


    def reset(self):
        # restart simulation
        libsumo.start(['-c', self.sumocfg])
        self.vehicle_ids = {id_: 1 for id_ in libsumo.vehicle.getIDList()}
        return self._grid()

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode='rgb_array'):
        """ basic rendering as there is no interaction possible with GUI in libsumo yet.
            supports simple lanes, junctions, traffic lights and cars
        """
        SCALE = 3
        LANE_COLOR = (128, 128, 128)
        JUNCTION_COLOR = (160, 160, 160)
        TL_COLORS = {'r': (0,0,255),
                     'y': (0,255,255),
                     'g': (0,128,0),
                     'G': (0,255,0)}
        CAR_COLOR = (0,160,160)
        bounds = np.array(libsumo.simulation.getNetBoundary())
        size = np.abs(np.diff(bounds, axis=0)).astype(int).flatten()*SCALE
        img = np.ones((size[1], size[0], 3), dtype=np.uint8)*255
        # draw each lane
        for lane in libsumo.lane.getIDList():
            shape = np.array(libsumo.lane.getShape(lane)) - bounds[0]
            shape = (shape*SCALE).astype(np.int32)
            # lane are lines, not polygons to by filled
            cv2.polylines(img, [shape], False, LANE_COLOR)
        # draw each junction
        for junc in libsumo.junction.getIDList():
            shape = np.array(libsumo.junction.getShape(junc))
            # junctions don't necessarily have coords
            if shape.shape[0] > 0:
                shape -= bounds[0]
                shape = (shape*SCALE).astype(np.int32)
                cv2.polylines(img, [shape], False, JUNCTION_COLOR)
        # draw the traffic lights
        for ligh in self.light_ids:
            ryg = libsumo.trafficlight.getRedYellowGreenState(ligh)
            # TODO assume traffic light only controls one junction
            junc = libsumo.trafficlight.getControlledJunctions(ligh)[0]
            junc_shape = libsumo.junction.getShape(junc)
            lanes = libsumo.trafficlight.getControlledLanes(ligh)
            # ignore lanes on same road, a lane has a an id "ROADID_LANE"
            roads = [l[:-2] for l in lanes]
            # each road will be intersected with the junction to get position of light
            l_unique, l_pos, l_count = np.unique(roads, return_index=True, return_counts=True)
            for i in range(len(l_unique)):
                segments = l_count[i]
                lane_shape = libsumo.lane.getShape(lanes[l_pos[i]])
                # combine each coord of lane with each coord of junction, compute their distance
                xi, yi = np.mgrid[slice(len(lane_shape)), slice(len(junc_shape))]
                diff = np.array(lane_shape)[xi] - np.array(junc_shape)[yi]
                dist = np.linalg.norm(diff.T.reshape(2,-1), axis=0)
                # junction coord will be one with smallest distance with lane
                intersection_i = np.argmin(dist)//len(yi)
                # breakpoint()
                line = np.array(junc_shape+(junc_shape[0],))[intersection_i:intersection_i+2]
                line = line[:2] - bounds[0]
                # split that line in number of lights for that lane
                delta = (line[1]-line[0])/2/(segments-1)
                lane_light_pos = line[0] + delta.reshape(1,2)*np.arange(segments).reshape(-1,1)
                lane_light_pos = (lane_light_pos*SCALE).astype(np.int32)
                l_colors = ryg[l_pos[i]:l_pos[i]+segments]
                # draw light as cirlce on lane
                for j in range(len(lane_light_pos)):
                    cv2.circle(img, tuple(lane_light_pos[j]), 2, TL_COLORS[l_colors[j]], thickness=-1)

        # draw each vehicle
        for veh in self.vehicle_ids.keys():
            pos = np.array(libsumo.vehicle.getPosition(veh)) - bounds[0]
            angle = -np.radians(libsumo.vehicle.getAngle(veh))
            # car shape is triangle
            shape = np.array([[-1,-2],[0,2],[1,-2]])
            # rotate car to appropriate angle
            R = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            shape = R.dot(shape.T).T+pos
            shape = (shape*SCALE).astype(np.int32)
            cv2.fillPoly(img, [shape], CAR_COLOR)
        
        return img



class TrafficLightSumoEnv(SumoEnv):

    def reset(self):
        self.total_waiting = 0
        self.max_waiting = 0
        return super(TrafficLightSumoEnv, self).reset()

    def step(self, action):
        # TODO for now assume single traffic light
        action = [action]
        # TODO for now assume traffic lights with 4 phases (RG->RY->GR->YR)
        # action 0: stays in same phase, action 1: go to next phase
        for tli in range(len(self.light_ids)):
            l = self.light_ids[tli]
            # if light is yellow, in-between phase, don't perform action
            if 'y' not in libsumo.trafficlight.getRedYellowGreenState(l):
                p = libsumo.trafficlight.getPhase(l)
                if action[tli] == 1:
                    # next phase
                    p = (p+1)%4
                # setting the same phase resets the duration
                libsumo.trafficlight.setPhase(l, p)

        # TODO multiple steps at once?
        libsumo.simulationStep()
        # update vehicle states
        reached_destination = self._vehicles_delta()
        v = np.array(list(self.vehicle_ids.values()))
        total_waiting = np.sum(v)
        max_waiting = np.std(v[np.nonzero(v)])

        total_wait_diff = self.total_waiting - total_waiting
        max_waiting_diff = self.max_waiting - max_waiting
        
        self.total_waiting = total_waiting
        self.max_waiting = max_waiting

        return self._grid(), np.array([reached_destination, -max_waiting/10]), False, {}

