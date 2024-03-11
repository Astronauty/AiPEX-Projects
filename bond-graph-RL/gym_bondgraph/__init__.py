from gymnasium.envs.registration import register

register(
    id='gym_bondgraph/BondGraphEnv-v4',
    entry_point='gym_bondgraph.envs:BondGraphEnv',
    max_episode_steps=300
)