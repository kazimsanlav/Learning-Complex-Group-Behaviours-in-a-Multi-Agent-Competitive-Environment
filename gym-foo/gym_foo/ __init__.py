from gym.envs.registration import register

register(
    id='fish_swarm-v0',
    entry_point='gym_foo.envs:FooEnv',
)
