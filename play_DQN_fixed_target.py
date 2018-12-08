
import random
import gym
from gym import wrappers
import make_env_
import numpy as np
import csv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

#^ Set parameters

env = make_env_.make_env('swarm',benchmark=True)

num_of_agents = env.n

state_size = (2+2+2*(num_of_agents-1)*2) # [agent's velocity(2d vector) + agent's position(2d vector) + 
                # other agent's relative position((n-1)*2d vector) + 
                # other agent's relative velocity((n-1)*2d vector)) ] 
                # in 3 agent case it is 2+2+2*2+2*2=12
                
action_size = 4 # discrete action space [up,down,left,right]

batch_size = 32 # used for batch gradient descent update

testing = False # render or not, expodation vs. exploration

n_episodes = 100000 if not testing else 100 # number of simulations 
n_steps = 100 if not testing else 300 # number of steps

load_episode = 1000 

updating_target_freq = 50 # rate C, reset W` <- W

output_dir = 'model_output/swarm/DQQ_fixed_target_10v1'

# ────────────────────────────────────────────────────────────────────────────────
# if testing:
#    env = wrappers.Monitor(env,(output_dir+'/movies'), force= True) #save as mp4
# ────────────────────────────────────────────────────────────────────────────────


#^ Define agent

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # defined above
        self.action_size = action_size # defined above
        self.memory = deque(maxlen=2000) # double-ended queue; removes the oldest element each time that you add a new element.
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 if not testing else 0.1 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = (1-0.0005) # exponential decay rate for exploration prob
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.0005 # learning rate of NN
        self.evaluation_model = self._build_model()  
        self.target_model = self._build_model()  
    
    def _build_model(self):
        # neural net for approximating Q-value function: Q*(s,a) ~ Q(s,a;W)
        model = Sequential() #fully connected NN
        model.add(Dense(state_size*2, input_dim=self.state_size, activation='relu')) # 1st hidden layer
        model.add(Dense(state_size*2, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 4 actions, so 4 output neurons
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_weights(self):
        self.target_model.set_weights(self.evaluation_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # take random action with epsilon probability
            
            onehot_action = np.zeros(action_size+1)
            onehot_action[random.randint(1,4)] = 1
            return onehot_action

        act_values = self.evaluation_model.predict(state) # predict reward value based on current state
        # print(act_values)
        act_index = np.argmax(act_values[0]) # pick the action with highest value

        onehot_action = np.zeros(action_size+1)
        onehot_action[act_index+1] = 1
        return onehot_action

   
    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done then target = reward
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model            
            next_state = np.reshape(next_state, [1, state_size]) #! reshape the state for DQN model
            
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.target_model.predict(next_state))) # (maximum target Q based on future action a')
            
            target_f = self.evaluation_model.predict(state) # approximately map current state to future discounted reward
            target_f[0][np.argmax(action)-1] = target
            history = self.evaluation_model.fit(state, target_f, epochs=1, verbose=0) 
            # single epoch of training with x=state
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def load(self, name):
        self.evaluation_model.load_weights(name)
        self.update_target_weights()

    def save(self, name):
        self.evaluation_model.save_weights(name)

#^ Interact with environment

agents = [ DQNAgent(state_size, action_size) for agent in range(num_of_agents) ] # initialise agents

#! create model output folders
for i,agent in enumerate(agents):
    if not os.path.exists(output_dir + "/weights/agent{}".format(i)):
        os.makedirs(output_dir + "/weights/agent{}".format(i))

#! load weights if exist    
load_episode = 100
for i,agent in enumerate(agents):
    file_name = (output_dir + "/weights/agent{}/".format(i) +"weights_" + '{:04d}'.format(load_episode) + ".hdf5")
    if os.path.isfile(file_name):
        print("Loading of {} model weights to use for agent {}".format(i))
        agent.load(file_name)

#! statistics
# ────────────────────────────────────────────────────────────────────────────────
collision_ = ['collision_{}'.format(i) for i in range(num_of_agents)]
loss_ = ['loss_{}'.format(i) for i in range(num_of_agents)]
reward_ = ['reward_{}'.format(i) for i in range(num_of_agents)]
statistics = ['episode','epsilon']+collision_+reward_+loss_

if not testing:
    with open(output_dir + '/statistics.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(statistics)
    csvFile.close()
# ────────────────────────────────────────────────────────────────────────────────

for episode in range(1,n_episodes+1): # iterate over new episodes of the game
    if(episode % 500 == 0): 
        n_steps+=50
        updating_target_freq+=25
    # ────────────────────────────────────────────────────────────────────────────────
    #^ for statistics
    statictics_row=[]
    collisions = [0]*num_of_agents
    rewards = [0]*num_of_agents
    losses = [0]*num_of_agents
    # ────────────────────────────────────────────────────────────────────────────────

    states = env.reset() # reset states at start of each new episode of the game
   
    for step in range(1,n_steps+1): # for every step
    # ────────────────────────────────────────────────────────────────────────────────
        #! reset target model weights
        if(step % updating_target_freq == 0):
            for agent in agents:
                agent.update_target_weights()
    # ────────────────────────────────────────────────────────────────────────────────
        if (testing): env.render();
        # ─────────────────────────────────────────────────────────────────
        # if(episode > 100 and episode < 110): env.render();
        # if(episode > 500 and episode < 510): env.render();
        # if(episode > 950 and episode < 1000): env.render(); 
        # ─────────────────────────────────────────────────────────────────
        all_actions=[]
        for state,agent in zip(states,agents):
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model
            action_i = agent.act(state)
            all_actions.append(action_i)
        
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
            agent.remember(states[i], all_actions[i], rewards[i], next_states[i], dones[i]) 
            # remember the previous timestep's state, actions, reward vs.        
       
        states = next_states # update the states
    
    print("episode: {}/{}, collisions: {}, epsilon: {:.2}".format(episode, n_episodes, collisions[0], agent.epsilon))
    for i,agent in  enumerate(agents):
        if len(agent.memory) > batch_size:
            history = agent.replay(batch_size) # train the agent by replaying the experiences of the episode
            
            losses[i] += history.history['loss'][0]
    
    # ────────────────────────────────────────────────────────────────────────────────
    #* episode,epsilon,collisions,rewards,losses statistics written    
    statictics_row.append(episode)      
    statictics_row.append(agents[0].epsilon)
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
                file_name = (output_dir + "/weights/agent{}/".format(i) +"weights_" + '{:04d}'.format(episode) + ".hdf5")
                agent.save(file_name)

    