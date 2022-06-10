from gym.envs.registration import register

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