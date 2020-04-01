import numpy as np

pole_velocity_min_range = -1.0
pole_velocity_max_range = 1.0
pole_velocity_num_decimals = 1
num_velocity = int((pole_velocity_max_range - pole_velocity_min_range) * 10**pole_velocity_num_decimals + 1)

pole_angle_min_range = -0.2
pole_angle_max_range = 0.2
pole_angle_num_decimals = 2
num_pole = int((pole_angle_max_range - pole_angle_min_range) * 10**pole_angle_num_decimals + 1)

def obs_to_index(obs_v,min_v,max_v,num_dec):
    obs_v = np.round(obs_v,num_dec)
    obs_v = np.clip(obs_v,min_v,max_v)
    ret_index = int((obs_v - min_v) * 10**num_dec)
    return ret_index

def get_q_index(obs):
    global pole_velocity_min_range, pole_velocity_max_range, pole_velocity_num_decimals
    global pole_angle_min_range, pole_angle_max_range, pole_angle_num_decimals
    obs_vel_index = obs_to_index(obs[3],pole_velocity_min_range,pole_velocity_max_range,pole_velocity_num_decimals)
    obs_angle_index = obs_to_index(obs[2],pole_angle_min_range,pole_angle_max_range,pole_angle_num_decimals)
    return obs_vel_index, obs_angle_index

def gauss(x, mean, std):
    return np.exp(- ((x-mean)**2) / (2*std**2))

### REWARDS ###

# Original
def get_reward_1(obs, reward):
    '''rew1'''
    return reward

# Gaussian
def get_reward_2(obs, reward):
    '''rew2'''
    return gauss(obs[2], mean=0.0, std=0.03)

# Original reducted
def get_reward_3(obs, reward):
    '''rew3'''
    return 1.0 if ( obs[2]>-0.02 and obs[2]<0.02 ) else 0.0

# Original reducted
def get_reward_4(obs, reward):
    '''rew4'''
    return gauss(obs[2], mean=0.0, std=0.03) * gauss(obs[0], mean=0.0, std=0.4)

### STATES ###

# Original
def get_state_1(obs):
    '''st1'''
    return obs

# Half state
def get_state_2(obs):
    '''st2'''
    return obs[2:]