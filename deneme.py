import make_env_
import multiagent
import random
# creates an multiagent environment which has reset, render, step
env = make_env_.make_env('simple_tag')
for i_episode in range(15):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(len(env.action_space))
        # act_space = [[0, act_space.n - 1] for act_space in range(4)]

        action_n = [[0, random.random(), random.random(), random.random(), random.random()],
                    [0, random.random(), random.random(), random.random(), random.random()],
                    [0, random.random(), random.random(), random.random(), random.random()],
                    [0, random.random(), random.random(), random.random(), random.random()]]
        observation, reward, done, info = env.step(action_n)
env.close()

# import random
# print(random.sample([1,2,4,1,5,4],4))
