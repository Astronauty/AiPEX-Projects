from gymnasium.envs.registration import register

register(
    id='gym_bondgraph/BondGraphEnv-v4',
    entry_point='gym_bondgraph.envs:BondGraphEnv',
    max_episode_steps=1000
)

register(
    id='gym_bondgraph/BondGraphSuspEnv',
    entry_point = 'gym_bondgraph.envs:BondGraphSuspEnv',
    max_episode_steps = 800
)

register(
    id='gym_bondgraph/QuarterCarSuspEnv',
    entry_point = 'gym_bondgraph.envs:QuarterCarSuspEnv',
    max_episode_steps = 100
)


register(
    id='gym_bondgraph/HalfCarSuspEnv',
    entry_point = 'gym_bondgraph.envs:HalfCarSuspEnv',
    max_episode_steps = 5000
)