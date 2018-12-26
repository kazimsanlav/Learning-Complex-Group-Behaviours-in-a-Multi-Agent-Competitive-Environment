
import random
import gym
import make_env_
import numpy as np
import csv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

np.random.seed(1)

#^ Set parameters

env = make_env_.make_env('swarm',benchmark=True)

num_of_agents = env.n

state_size = (2+2+2*(num_of_agents-1)*2) # [agent's velocity(2d vector) + agent's position(2d vector) + 
                # other agent's relative position((n-1)*2d vector) + 
                # other agent's relative velocity((n-1)*2d vector)) ] 
                # in 3 agent case it is 2+2+2*2+2*2=12
                
action_size = 4 # discrete action space [up,down,left,right]

# batch_size = 32 # used for batch gradient descent update

testing = True # render or not, expodation vs. exploration

n_episodes = 20000 if not testing else 50 # number of simulations 
n_steps = 100 if not testing else 100 # number of steps

load_episode = 1

output_dir = 'model_output/swarm/DPG'

# # ────────────────────────────────────────────────────────────────────────────────
# if testing:
#     import pyautogui
#  ────────────────────────────────────────────────────────────────────────────────


#^ Define agent
class DPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # defined above
        self.action_size = action_size # defined above
        self.gamma = 0.95 # discount rate
        self.learning_rate = 0.001 if not testing else 0 # learning rate of NN
        self.observations = []
        self.actions = [] 
        self.rewards = []
        self.model = self._build_model()  
    
    def _build_model(self):
        # neural net for approximating Q-value function: Q*(s,a) ~ Q(s,a;W)
        model = Sequential() #fully connected NN
        model.add(Dense(state_size*2, input_dim=self.state_size, activation='relu')) # 1st hidden layer
        model.add(Dense(state_size*2, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='softmax')) # 4 actions, so 4 output neurons
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward):
        '''
        Add state,action,reward to the memory
        '''
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def act(self, state):
        # state = state.reshape([1, state.shape[0]])
        action_prob = self.model.predict(state, batch_size=1).flatten()
        act_index = np.random.choice(self.action_size, 1, p=action_prob)[0]
        onehot_action = np.zeros(action_size+1)
        onehot_action[act_index+1] = 1
        return onehot_action, action_prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        history = self.model.train_on_batch(X, Y)
        self.observations = []
        self.actions = []
        self.rewards = []
        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#^ Interact with environment

agents = [ DPGAgent(state_size, action_size) for agent in range(num_of_agents) ] # initialise agents

#! create model output folders
for i,agent in enumerate(agents):
    if not os.path.exists(output_dir + "/weights/agent{}".format(i)):
        os.makedirs(output_dir + "/weights/agent{}".format(i))

#! load weights if exist    
for i,agent in enumerate(agents):
    file_name = (output_dir + "/weights/agent{}/".format(i) +"weights_" + '{:04d}'.format(load_episode) + ".hdf5")
    if os.path.isfile(file_name):
        print("Loading weights to use for agent {}".format(i))
        agent.load(file_name)

#! statistics
# ────────────────────────────────────────────────────────────────────────────────
collision_ = ['collision_{}'.format(i) for i in range(num_of_agents)]
loss_ = ['loss_{}'.format(i) for i in range(num_of_agents)]
reward_ = ['reward_{}'.format(i) for i in range(num_of_agents)]
statistics = ['episode']+collision_+reward_+loss_

if not testing:
    with open(output_dir + '/statistics.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(statistics)
    csvFile.close()
# ────────────────────────────────────────────────────────────────────────────────

for episode in range(1,n_episodes+1): # iterate over new episodes of the game
    if(episode % 1000 == 0): n_steps+=50;
    # ────────────────────────────────────────────────────────────────────────────────
    #^ for statistics
    statictics_row=[]
    collisions = [0]*num_of_agents
    rewards = [0]*num_of_agents
    losses = [0]*num_of_agents
    # ────────────────────────────────────────────────────────────────────────────────

    states = env.reset() # reset states at start of each new episode of the game
    
    for step in range(1,n_steps+1): # for every step
        # env.render()

        if (testing): 
            env.render()
            # if (step % 4 == 0 ):
            #     # Take screenshot
            #     pic = pyautogui.screenshot()
            #     # Save the image
            #     pic.save(output_dir+'/screenshots/Screenshot_{}.png'.format(step)) 
        # ─────────────────────────────────────────────────────────────────
        # if(episode > 100 and episode < 110): env.render();
        # if(episode > 500 and episode < 510): env.render();
        # if(episode > 950 and episode < 1000): env.render();   
        # ─────────────────────────────────────────────────────────────────
        all_actions=[]
        all_probs=[]
        for state,agent in zip(states,agents):
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model
            action_i, prob_i = agent.act(state)
            all_actions.append(action_i)
            all_probs.append(prob_i)
        
        next_states, rewards, dones, infos = env.step(all_actions) # take a step (update all agents)

        # ─────────────────────────────────────────────────────────────────
        #* collision,reward statistics
        for i in range(num_of_agents):
            collisions[i] += (infos['collision'][i])
            rewards[i] += (rewards[i])
        # ────────────────────────────────────────────────────────────────────────────────

        for state in next_states:
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model

        for i,agent in enumerate(agents):
            agent.remember(states[i], all_actions[i], all_probs[i], rewards[i]) 
            # remember the previous timestep's state, actions, reward vs.        
       
        states = next_states # update the states
    
    print("\n episode: {}/{}, collisions: {}, rewards: {:.2f}|{:.2f}|{:.2f}".format(episode,
                                                                                    n_episodes,
                                                                                    collisions[0],
                                                                                    rewards[0],
                                                                                    rewards[1],
                                                                                    rewards[2]))

    for i,agent in enumerate(agents):
        history_i = agent.train()
        losses[i] += history_i

        # if len(agent.memory) > batch_size:
            # history = agent.replay(batch_size) # train the agent by replaying the experiences of the episode
            # losses[i] += history.history['loss'][0]
    
    # ────────────────────────────────────────────────────────────────────────────────
    #! episode,collisions,rewards,losses statistics written    
    statictics_row.append(episode)      
    statictics_row += (collisions)
    statictics_row += (rewards)
    statictics_row += (losses)

    if not testing:
        with open(output_dir + '/statistics.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(statictics_row)
        csvFile.close()

    # ────────────────────────────────────────────────────────────────────────────────
    #! save weights
    if not testing:
        if episode % 50 == 0:
            for i,agent in enumerate(agents):
                agent.save(output_dir + "/weights/agent{}/".format(i) +"weights_" + '{:04d}'.format(episode) + ".hdf5")

    