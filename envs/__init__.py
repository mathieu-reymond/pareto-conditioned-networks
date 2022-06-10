from envs.deep_sea_treasure import DeepSeaTreasureEnv, BountyfulSeaTreasureEnv, ConvexSeaTreasureEnv
from envs.minecart.env import MinecartEnv, MinecartDeterministicEnv, MinecartSimpleDeterministicEnv


from gym.envs.registration import register

# DST
register(
    id='DeepSeaTreasure-v0',
    entry_point='envs:DeepSeaTreasureEnv',
    )

# Minecart env
register(
    id='DeepSeaTreasure-v0',
    entry_point='envs:DeepSeaTreasureEnv',
    )

# SUMO env
register(
    id='CrossroadSumo-v0',
    entry_point='envs.gym_sumo.envs:CrossroadSumoEnv'
)

# WalkRoom env
register(
        id='Walkroom2D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 2}
    )

register(
        id='Walkroom3D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 3}
    )

register(
        id='Walkroom4D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 4}
    )

register(
        id='Walkroom5D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 20, 'dimensions': 5}
    )

register(
        id='Walkroom6D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 6}
    )

register(
        id='Walkroom7D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 7}
    )

register(
        id='Walkroom8D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 8}
    )

register(
        id='Walkroom9D-v0',
        entry_point='envs.walkroom.walkroom:WalkRoom',
        kwargs={'size': 10, 'dimensions': 9}
    )