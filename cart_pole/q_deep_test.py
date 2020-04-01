import gym
import tensorflow as tf
import numpy as np
# import trfl
from collections import deque

gym_env_name = 'CartPole-v1'
env = gym.make(gym_env_name)

train_name = 'rew4_st1_v0'

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(f'checkpoints/{train_name}/cartpole.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(f'./checkpoints/{train_name}/'))

nn_inputs = tf.get_default_graph().get_tensor_by_name('main/inputs:0')
nn_output = tf.get_default_graph().get_tensor_by_name('main/output/BiasAdd:0')

done = False
state = env.reset()

# while not done:
while True:
    feed = {nn_inputs: state.reshape((1, *state.shape))}
    Qs = sess.run(nn_output, feed_dict=feed)
    action = np.argmax(Qs)
    state, _, done, _ = env.step(action)
    env.render()