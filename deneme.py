import make_env_
from gym.spaces import prng
import numpy as np
import random

# creates an multiagent environment which has reset, render, step
env = make_env_.make_env('swarm') #! 2vs1 Swarm environment
# env = make_env_.make_env('simple_tag_guided_1v2')
# create interactive policies for each agent
# policies = [InteractivePolicy(env,i) for i in range(env.n)]
print(env.observation_space)
print(env.action_space)
# env.observation_space


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
    # print(action)
    return action

def fixed():
	act = [0,1,0,0]
	return act

for i_episode in range(1):# number of simulations 
    observation = env.reset()
    # print(len(observation[0]))
    for t in range(1):# number of steps
        # env.render()
	
        my_action = [ deterministic_actions()  for i in range(env.n) ]
        
        next_state, reward, done, info = env.step(my_action)
        print('*'*30)
        print('next_state\n',next_state)
        print('\nreward\n',reward)
        print('\ndone\n',done)
        print('\ninfo\n',info)
        print('*'*30)
        # print(len(observation)) 
        # print(observation[0].shape) 
        # print(observation[0]) 

env.close()


# obs = [np.array([ 0.        ,  0.        , -0.71170829,  0.82181501,  0.62097461, -0.89593826,  1.14709038,  0.16188317,  0.        ,  0.        ,0.        ,  0.        ]),
#  np.array([ 0.        ,  0.        , -0.09073367, -0.07412326, -0.62097461,  0.89593826,  0.52611577,  1.05782144,  0.        ,  0.        ,0.        ,  0.        ]), 
#  np.array([ 0.        ,  0.        ,  0.43538209,  0.98369818, -1.14709038, -0.16188317, -0.52611577, -1.05782144,  0.        ,  0.        ,0.        ,  0.        ])]