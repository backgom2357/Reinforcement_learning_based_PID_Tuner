import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time

import imageio

import sys
sys.path.append('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/')

from PID import PID
from A2C.a2c_agent import A2Cagent
cenv = gym.make("LunarLanderContinuous-v2")

SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

side_engine_control = 0
main_engine_control = 0
engine_off = False

human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global side_engine_control, main_engine_control, engine_off, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True  # Enter
    if key==32: human_sets_pause = not human_sets_pause # Space
    a = int(key)
    if a <= 0: return
    if a==ord('a'):
        side_engine_control = 0.3
    if a==ord('d'):
        side_engine_control = -0.3
    if a==ord('s'):
        main_engine_control = -0.3
    if a==ord('w'):
        main_engine_control = 0.3
    if a==ord('q'):
        engine_off=True

def key_release(key, mod):
    global side_engine_control, main_engine_control, engine_off
    if side_engine_control == -0.3 or side_engine_control == 0.3:
        side_engine_control = 0
    if main_engine_control == -0.3 or main_engine_control == 0.3:
        main_engine_control = 0
    if engine_off == True:
        engine_off = False

cenv.render()
cenv.unwrapped.viewer.window.on_key_press = key_press
cenv.unwrapped.viewer.window.on_key_release = key_release

class Connector(object):

    def __init__(self):
        self.observation_space = (5)
        self.action_space = (1)
        self.action_bound = (2)
    
    def reward_function(self, x):
        return 1 if abs(x) < 0.1 else -1

def rollout(env):
    global side_engine_control, main_engine_control, engine_off, human_wants_restart, human_sets_pause
    
    human_wants_restart = False
    
    state = cenv.reset()
    done = False
    action = [0,0]

    # Init Tuner
    connector = Connector()
    ## Angular Tuner
    a_tuner = A2Cagent(connector)
    a_state = np.zeros((1,connector.observation_space))
    a_p = a_tuner.actor.get_action(a_state)
    a_tuner.critic.model(a_state)
    # a_tuner.actor.load_weights('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/A2C/save_weights/tunner_actor499.h5')
    # a_tuner.critic.load_weights('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/A2C/save_weights/tunner_critic499.h5')
    a_batch_state, a_batch_action, a_batch_td_target, a_batch_advantage = [], [], [] ,[]

    ## Vertical Tuner
    v_tuner = A2Cagent(connector)
    v_state = np.zeros((1,connector.observation_space))
    v_p = v_tuner.actor.get_action(v_state)
    v_tuner.critic.model(v_state)
    # v_tuner.actor.load_weights('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/A2C/save_weights/tunner_actor499.h5')
    # v_tuner.critic.load_weights('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/A2C/save_weights/tunner_critic499.h5')
    v_batch_state, v_batch_action, v_batch_td_target, v_batch_advantage = [], [], [] ,[]


    # Init PID controllor
    apid = PID(a_p, 0, 0)
    vpid = PID(v_p, 0, 0)
    apid.SetPoint=0
    vpid.SetPoint=0

    cr = 0
    
    apid_sp_list, apid_fb_list = [], []
    vpid_sp_list, vpid_fb_list = [], []

    # images = []

    while not done:
     
        # window_still_open = cenv.render(mode='rgb_array')
        # images.append(np.array(window_still_open))
        window_still_open = cenv.render()
        n_state, reward, done, info = cenv.step(action)
        
        # PID control
        ## Tuner
        a_p = a_tuner.actor.get_action(a_state)
        v_p = v_tuner.actor.get_action(v_state)

        apid.Kp = a_p[0]
        vpid.Kp = v_p[0]

        apid.SetPoint=side_engine_control
        vpid.SetPoint=main_engine_control
        
        apid_sp_list.append(apid.SetPoint)
        vpid_sp_list.append(vpid.SetPoint)

        ## Control
        apid.update(n_state[4])
        action[1] = (-apid.output*2.5 + n_state[5])*20

        vpid.update(n_state[3])
        action[0] = vpid.output

        time.sleep(0.01)
        
        apid_fb_list.append(n_state[4])
        vpid_fb_list.append(n_state[3])

        # State
        a_state = np.reshape(a_state, (1, connector.observation_space))
        v_state = np.reshape(v_state, (1, connector.observation_space))

        # Next state
        a_next_state = [apid.SetPoint, round(n_state[4],3), round(apid.last_error,3), round(apid.ITerm), round(apid.Kp,3)]
        v_next_state = [vpid.SetPoint, round(n_state[3],3), round(vpid.last_error,3), round(vpid.ITerm), round(vpid.Kp,3)]
        a_next_state = np.reshape(a_state, (1, connector.observation_space))
        v_next_state = np.reshape(v_state, (1, connector.observation_space))

        # Reward for updating tuners
        a_reward = np.reshape(connector.reward_function(apid.last_error), [1,1])
        v_reward = np.reshape(connector.reward_function(vpid.last_error), [1,1])

        # Reshape actions of tuners
        a_p = np.reshape(a_p, [1, connector.action_space])
        v_p = np.reshape(v_p, [1, connector.action_space])

        # Calulate critics
        a_v_value = a_tuner.critic.model(a_state)
        a_next_v_value = a_tuner.critic.model(a_next_state)
        v_v_value = v_tuner.critic.model(v_state)
        v_next_v_value = v_tuner.critic.model(v_next_state)

        # Calculate advantages and TD targets
        a_adv, a_tg = a_tuner.advantage_td_target(a_reward, a_v_value, a_next_v_value, done)
        v_adv, v_tg = v_tuner.advantage_td_target(v_reward, v_v_value, v_next_v_value, done)

        # Append to batch
        a_batch_state.append(a_state)
        a_batch_action.append(a_p)
        a_batch_td_target.append(a_tg)
        a_batch_advantage.append(a_adv)

        v_batch_state.append(v_state)
        v_batch_action.append(v_p)
        v_batch_td_target.append(v_tg)
        v_batch_advantage.append(v_adv)

        if engine_off:
            action = [0, 0]

        print(f'action:[{action[0]:+0.2f},{action[1]:+0.2f}], as:{n_state[5]:+0.3f},vs:{n_state[3]:+0.3f}, vsp:{vpid.SetPoint:+0.1f}, asp:{apid.SetPoint:+0.1f}', end='\r')
        cr+=reward
        
        if window_still_open==False: return False
        if human_wants_restart: break
        while human_sets_pause:
            cenv.render()
            time.sleep(0.1)

        if len(a_batch_state) < 8:
            a_state = a_next_state[0]
            v_state = v_next_state[0]
            continue
        
        # Online Train
        ## A_tuner update
        a_states = a_tuner.unpack_batch(a_batch_state)
        a_actions = a_tuner.unpack_batch(a_batch_action)
        a_td_targets = a_tuner.unpack_batch(a_batch_td_target)
        a_advantages = a_tuner.unpack_batch(a_batch_advantage)
        ## Clear batch
        a_batch_state, a_batch_action, a_batch_td_target, a_batch_advantage = [], [], [] ,[]
        # Critic neural net update
        a_tuner.critic.train_on_batch(a_states, a_td_targets)
        # Actor neural net update
        a_tuner.actor.train(a_states, a_actions, a_advantages)
        
        ## V_tuner update
        v_states = v_tuner.unpack_batch(v_batch_state)
        v_actions = v_tuner.unpack_batch(v_batch_action)
        v_td_targets = v_tuner.unpack_batch(v_batch_td_target)
        v_advantages = v_tuner.unpack_batch(v_batch_advantage)
        ## Clear batch
        v_batch_state, v_batch_action, v_batch_td_target, v_batch_advantage = [], [], [] ,[]
        # Critic neural net update
        v_tuner.critic.train_on_batch(v_states, v_td_targets)
        # Actor neural net update
        v_tuner.actor.train(v_states, v_actions, v_advantages)

        # Update state
        a_state = a_next_state[0]
        v_state = v_next_state[0]

    print()
    print("reward:",cr)

    ## For Collecting Data
    # imageio.mimsave('tuner_applied.gif', images, duration=1/60)
    # np.save('apid_sp_list.npy', apid_sp_list)
    # np.save('apid_fb_list.npy', apid_fb_list)
    # np.save('vpid_sp_list.npy', vpid_sp_list)
    # np.save('vpid_fb_list.npy', vpid_fb_list)

    return

while 1:
    window_still_open = rollout(cenv)
    if window_still_open==False: break