from gym_sumo.envs.sumo import TrafficLightSumoEnv
import os


def config_path(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'configs', filename)


def CrossroadSumoEnv():
    return TrafficLightSumoEnv(config_path('crossroad/crossroad.sumocfg'))