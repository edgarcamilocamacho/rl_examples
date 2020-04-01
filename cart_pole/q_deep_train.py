import gym
import tensorflow as tf
import numpy as np
from collections import deque

from learning_curve import LearningCurve
from q_table_common import *

reward_func = get_reward_1
state_func = get_state_1

if state_func==get_state_1:
    state_size = 4
else:
    state_size = 2

training_name = f'{reward_func.__doc__}_{state_func.__doc__}'

gym_env_name = 'CartPole-v0'
env = gym.make(gym_env_name)

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, 
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size, activation_fn=tf.nn.relu)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size, activation_fn=tf.nn.relu)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None, scope='output')
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

# Training
train_episodes = 900
epsilon_episodes = 800
test_episodes = 100

# Exploration
start_epsilon = 1.0
min_epsilon = 0.3

# Q Learning Parameters
alpha = 0.0001
gamma = 0.99
hidden_size = 64

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 200                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

# Metrics
utility_list = [] 
avg_window = 10
view_interval = 1000

tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=alpha, state_size=state_size)

# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
obs, rew, done, _ = env.step(env.action_space.sample())
reward = reward_func(obs, rew)
state = state_func(obs)

memory = Memory(max_size=memory_size)

if gym_env_name == 'CartPole-v0':
    max_util = 200
else:
    max_util = 500

curve = LearningCurve(  plots=[ ('utility', 'left', 'r'), 
                                ('epsilon','right', 'b')],
                        episode_range=1000,
                        min_y_left = 0, 
                        max_y_left = max_util)
                        
# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    reward = reward_func(obs, rew)
    next_state = state_func(obs)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        
        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        obs, rew, done, _ = env.step(env.action_space.sample())
        reward = reward_func(obs, rew)
        state = state_func(obs)
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

# Now train with experiences
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    step = 0
    for episode in range(1, train_episodes+test_episodes):
        if episode<=epsilon_episodes:
            epsilon = episode*(min_epsilon - start_epsilon)/epsilon_episodes + start_epsilon
        elif episode<=train_episodes:
            epsilon = min_epsilon
        else:
            epsilon = 0.0

        done = False
        utility = 0

        # Start new episode
        obs = env.reset()
        state = state_func(obs)
        
        while not done:
            if np.random.rand() <= epsilon:
                # Make a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)
            
            obs, rew, done, _ = env.step(action)
            reward = reward_func(obs, rew)
            next_state = state_func(obs)
            utility += reward
            
            if done:
                next_state = np.zeros(state.shape)
                
            memory.add((state, action, reward, next_state))
            state = next_state

            if episode<=train_episodes:
            
                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                
                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
                
                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = (0, 0)
                
                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                            mainQN.targetQs_: targets,
                                            mainQN.actions_: actions})
        
        utility_list.append(utility)
        if len(utility_list) == avg_window:
            curve.add_sample( ['utility', 'epsilon'], 
                            episode, 
                            [np.mean(utility_list), epsilon])    
            utility_list = []
        
    saver.save(sess, f'checkpoints/{training_name}_{gym_env_name[-2:]}/cartpole.ckpt')
    curve.save_plot(f'./curves/q_deep_{training_name}_{gym_env_name[-2:]}.png')
