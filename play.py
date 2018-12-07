
import random
import gym
import make_env_
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

#^ Set parameters

env = make_env_.make_env('swarm')

num_of_agents = env.n

state_size = 12 # [agent's velocity(2d vector) + agent's position(2d vector) + 
                # other agent's relative position((n-1)*2d vector) + 
                # other agent's relative velocity((n-1)*2d vector)) ] 
                # in 3 agent case it is 2+2+2*2+2*2=12
                
action_size = 4 # discrete action space [up,down,left,right]

batch_size = 32 # used for mini-batch gradient descent update

n_episodes = 1001 # number of simulations 

output_dir = 'model_output/swarm/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#^ Define agent

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # defined above
        self.action_size = action_size # defined above
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 #! WHAT IS THIS
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters 
        self.model = self._build_model()  
    
    def _build_model(self):
        # neural net for approximating Q-value function: Q*(s,a) ~ Q(s,a;W)
        model = Sequential() #fully connected NN
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 4 actions, so 4 output neurons
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        
        #! commented
        # if np.random.rand() <= self.epsilon: # take random action with epsilon probability
            
        #     onehot_action = np.zeros(action_size+1)
        #     onehot_action[random.randint(1,4)] = 1
        #     return onehot_action

        act_values = self.model.predict(state) # predict reward value based on current state
        # print(act_values)
        act_index = np.argmax(act_values[0]) # pick the action with highest value

        onehot_action = np.zeros(action_size+1)
        onehot_action[act_index+1] = 1
        return onehot_action

    #! NEED MODIFICATION
    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model            
            next_state = np.reshape(next_state, [1, state_size]) #! reshape the state for DQN model
            
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state))) # (maximum target Q based on future action a')
            
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][np.argmax(action)-1] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
       
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#^ Interact with environment

agents = [ DQNAgent(state_size, action_size) for agent in range(num_of_agents) ] # initialise agents

done = False

for e in range(n_episodes): # iterate over new episodes of the game
    states = env.reset() # reset states at start of each new episode of the game
    
    for step in range(100): # for every step
        env.render()
        
        all_actions=[]
        for state,agent in zip(states,agents):
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model
            action_i = agent.act(state)
            all_actions.append(action_i)
        
        print(all_actions)

        next_states, rewards, done, info = env.step(all_actions) # take a step (update all agents)

        for state in next_states:
            state = np.reshape(state, [1, state_size]) #! reshape the state for DQN model

        for i,agent in enumerate(agents):
            agent.remember(states[i], all_actions[i], rewards[i], next_states[i], done[i]) # remember the previous timestep's state, actions, reward vs.        
       
        states = next_states # update the states
    
    print("episode: {}/{}, step: {}, epsilon: {:.2}".format(e, n_episodes, step, agent.epsilon))
    for agent in agents:
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        
    # if e % 50 == 0:
    #     agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5"

