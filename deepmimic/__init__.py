from gym.envs.registration import register
register(
    id='heygogo-v0',
    entry_point='deepmimic.envs:hmyifEnv'
)