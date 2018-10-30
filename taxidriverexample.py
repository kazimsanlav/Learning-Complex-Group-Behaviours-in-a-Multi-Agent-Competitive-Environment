
# ^ Creating the agent
# * First thing to do is import all the libraries we’ll need: 

import gym 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline
 
# For animation 
from IPython.display import clear_output
from time import sleep

"""
 The idea is that we can specify a single parameter, method, and use that to 
 determine which algorithm the agent uses to learn.
"""
class Agent: 
    
    def __init__(self, method, start_alpha = 0.3, start_gamma = 0.9, start_epsilon = 0.5):
        self.method = method
        self.env = gym.make('Taxi-v2')
        self.n_squares = 25 
        self.n_passenger_locs = 5 
        self.n_dropoffs = 4 
        self.n_actions = self.env.action_space.n
        self.epsilon = start_epsilon
        self.gamma = start_gamma
        self.alpha = start_alpha
        # Set up initial q-table 
        self.q = np.zeros(shape = (self.n_squares*self.n_passenger_locs*self.n_dropoffs, self.env.action_space.n))
        # Set up policy pi, init as equiprobable random policy
        self.pi = np.zeros_like(self.q)
        for i in range(self.pi.shape[0]): 
            for a in range(self.n_actions): 
                self.pi[i,a] = 1/self.n_actions
    
    def simulate_episode(self):
        s = self.env.reset()
        done = False
        r_sum = 0 
        n_steps = 0 
        gam = self.gamma
        while not done: 
            n_steps += 1
            # take action from policy
            x = np.random.random()
            a = np.argmax(np.cumsum(self.pi[s,:]) > x) 
            # take step 
            s_prime,r,done,info = self.env.step(a)    
            if self.method == 'q_learning': 
                a_prime = np.random.choice(np.where(self.q[s_prime] == max(self.q[s_prime]))[0])
                self.q[s,a] = self.q[s,a] + self.alpha * \
                    (r + gam*self.q[s_prime,a_prime] - self.q[s,a])
            elif self.method == 'sarsa': 
                a_prime = np.argmax(np.cumsum(self.pi[s_prime,:]) > np.random.random())
                self.q[s,a] = self.q[s,a] + self.alpha * \
                    (r + gam*self.q[s_prime,a_prime ] - self.q[s,a])
            elif self.method == 'expected_sarsa':
                self.q[s,a] = self.q[s,a] + self.alpha * \
                    (r + gam* np.dot(self.pi[s_prime,:],self.q[s_prime,:]) - self.q[s,a])
            else: 
                raise Exception("Invalid method provided")
            # update policy
            best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
            for i in range(self.n_actions): 
                if i == best_a:      self.pi[s,i] = 1 - (self.n_actions-1)*(self.epsilon / self.n_actions)
                else:                self.pi[s,i] = self.epsilon / self.n_actions

            # decay gamma close to the end of the episode
            if n_steps > 185: 
                gam *= 0.875
            s = s_prime
            r_sum += r
        return r_sum




"""
# ^ Training the agent
Here’s the code for training the agent, where we run lots of episodes in which the 
agent attempts to maximise its reward.
"""
def train_agent(agent, n_episodes= 100001, epsilon_decay = 0.99995, alpha_decay = 0.99995, print_trace = False):
    r_sums = []
    for ep in range(n_episodes): 
        r_sum = agent.simulate_episode()
        # decrease epsilon and learning rate 
        agent.epsilon *= epsilon_decay
        agent.alpha *= alpha_decay
        if print_trace: 
            if ep % 20000 == 0 and ep > 0 : 
                print("Episode:", ep, "alpha:", np.round(agent.alpha, 3), "epsilon:",  np.round(agent.epsilon, 3))
                print ("Last 100 episodes avg reward: ", np.mean(r_sums[ep-100:ep]))
        r_sums.append(r_sum)
    return r_sums

"""
# ? Decreasing epsilon (our exploration parameter) over time makes sense.
# ? Each episode, the agent becomes more and more confident what good and bad choices look like. 
# ? Decreasing epsilon is required if we seek to maximise our reward sum,
# ? since exploratory actions typically aren’t optimal actions, and besides, 
# ? we can be fairly sure what are good actions and bad actions by this point anyway. 
# ? There is a similar argument for decreasing alpha (the learning rate) over time –
# ? we don’t want the estimates to jump around too much once we are confident in their validity.
"""

# * Now we can create our agents and train them.
# Create agents 
sarsa_agent = Agent(method='sarsa')
e_sarsa_agent = Agent(method='expected_sarsa')
q_learning_agent = Agent(method='q_learning')
 
# Train agents
r_sums_sarsa = train_agent(sarsa_agent, print_trace=True)
r_sums_e_sarsa = train_agent(e_sarsa_agent, print_trace=True)
r_sums_q_learning = train_agent(q_learning_agent, print_trace=True)


"""
# ^ Which method is best?

After training our agents we can compare their performance to each other. 
The criteria for comparison we’ll use is the best 100-epsiode average reward for each agent.

Let’s plot the 100-epsiode rolling cumulative reward over time:
"""
df = pd.DataFrame({"Sarsa": r_sums_sarsa, 
             "Expected_Sarsa": r_sums_e_sarsa, 
             "Q-Learning": r_sums_q_learning})
df_ma = df.rolling(100, min_periods = 100).mean()
df_ma.iloc[1:1000].plot()
plt.show()

# ^Viewing the policy
def generate_frames(agent):
    start_state = agent.env.reset()
    agent.env.env.s = start_state
    s = start_state
    policy = np.argmax(agent.pi,axis =1)
    epochs = 0
    penalties, reward = 0, 0
    frames = [] 
    done = False
    frames.append({
        'frame': agent.env.render(mode='ansi'),
        'state': agent.env.env.s ,
        'action': "Start",
        'reward': 0
        }
    )
    while not done:
        a = policy[s]
        s, reward, done, info = agent.env.step(a)
        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': agent.env.render(mode='ansi'),
            'state': s,
            'action': a,
            'reward': reward
            }
        )
        epochs += 1
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
    return frames

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.4)

print_frames(generate_frames(sarsa_agent))
# ^Conclusion
"""
The three agents seemed to perform around the same on this task, 
with Sarsa being a little worse than the other two. This result doesn’t always hold –
on some tasks (see “The Cliff” – Sutton and Barto (2018)) they perform very differently, 
but here the results were similar. It would be interesting to see how a planning 
/ n-step algorithm would perform on this task.
"""