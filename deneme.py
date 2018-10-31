import make_env_

# creates an multiagent environment which has reset, render, step
env = make_env_.make_env('simple_tag')
# create interactive policies for each agent
# policies = [InteractivePolicy(env,i) for i in range(env.n)]

for i_episode in range(15):
    observation = env.reset()
    for t in range(100):
        env.render()
        

        # my_action0 = [ 
        #     0,
        #     env.action_space[0].sample(),
        #     env.action_space[1].sample(),
        #     env.action_space[2].sample(),
        #     env.action_space[3].sample()
        #     ]

        def sample_actions():
            action = [ 
            0,
            env.action_space[0].sample(),
            env.action_space[1].sample(),
            env.action_space[2].sample(),
            env.action_space[3].sample()
            ]
            return action

        my_action = [ sample_actions(), sample_actions(), sample_actions(), sample_actions() ]

        
        # my_action = [ my_action0, my_action0, my_action0, my_action0 ]

        # print(my_action)
        
        observation, reward, done, info = env.step(my_action)




env.close()


