from gym.envs.multi_objective.minecart.minecart import Minecart
import os


def config_path(filename):
    return os.path.join(os.path.dirname(__file__), 'configs', filename)

def MinecartEnv():
    return Minecart.from_json(config_path('mine_config.json'))


def MinecartDeterministicEnv():
    return Minecart.from_json(config_path('mine_config_det.json'))


def MinecartSimpleDeterministicEnv():
    return Minecart.from_json(config_path('mine_config_1ore_det.json'))
