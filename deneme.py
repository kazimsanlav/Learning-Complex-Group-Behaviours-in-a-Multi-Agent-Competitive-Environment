import make_env_
from gym.spaces import prng
import numpy as np
import random

# creates an multiagent environment which has reset, render, step
env = make_env_.make_env('swarm') #! my enviorenment
# env = make_env_.make_env('simple_tag_guided_1v2')
# create interactive policies for each agent
# policies = [InteractivePolicy(env,i) for i in range(env.n)]
print(env.observation_space)
print(env.action_space)
# state_size = env.observation_space.shape[0]
# state_size

# exit()
# print(policies)
# exit()
def sample_actions():
    action = [env.action_space[i].sample() for i in range(env.n)]
    action.insert(0,0)
    # print(action)
    return action

def deterministic_actions():
    action = fixed() 
    action.insert(0,0)
    print(action)
    return action

def fixed():
	act = [0,0,0,1]
	return act

# def genetic_actions():	
# 	action = []
# 	for adv in env.agents:
# 		if adv.advesary:
# 			action.append([1.0,0])
# 		else:
# 			action.append([-1.0,0])
# 	action.insert(0,0)
# 	return action

for i_episode in range(5):# number of simulations 
    observation = env.reset()
    for t in range(100):# number of steps
        env.render()
	
        my_action = [ sample_actions()  for i in range(env.n) ]
        
        observation, reward, done, info = env.step(my_action)

        # print(reward)

# genetic_actions()
env.close()


