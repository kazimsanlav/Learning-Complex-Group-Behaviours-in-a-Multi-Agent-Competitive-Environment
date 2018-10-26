import gym
env = gym.make('Hopper-v2')
env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # take a random action