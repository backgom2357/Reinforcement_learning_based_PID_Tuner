import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time

from PID import PID

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

def rollout(env):
    global side_engine_control, main_engine_control, engine_off, human_wants_restart, human_sets_pause
    
    human_wants_restart = False
    
    state = cenv.reset()
    done = False
    action = [0,0]

    apid = PID(1.253136190601821, 0.620026107056804, -0.002720052821954503)
    vpid = PID(1.253136190601821, 0.620026107056804, -0.002720052821954503)
    apid.SetPoint=0
    vpid.SetPoint=0

    cr = 0
    
    apid_sp_list, apid_fb_list = [], []
    vpid_sp_list, vpid_fb_list = [], []

    while not done:
     
        window_still_open = cenv.render()
        n_state, reward, done, info = cenv.step(action)
        
        # PID control
        apid.SetPoint=side_engine_control
        vpid.SetPoint=main_engine_control
        
        apid_sp_list.append(apid.SetPoint)
        vpid_sp_list.append(vpid.SetPoint)

        apid.update(n_state[4])
        action[1] = (-apid.output*2.5 + n_state[5])*20

        vpid.update(n_state[3])
        action[0] = vpid.output
        
        apid_fb_list.append(n_state[4])
        vpid_fb_list.append(n_state[3])

        time.sleep(0.01)

        if engine_off:
            action = [0, 0]

        print(f'action:[{action[0]:+0.2f},{action[1]:+0.2f}], as:{n_state[5]:+0.3f},vs:{n_state[3]:+0.3f}, vsp:{vpid.SetPoint:+0.1f}, asp:{apid.SetPoint:+0.1f}', end='\r')
        cr+=reward
        
        if window_still_open==False: return False
        if human_wants_restart: break
        while human_sets_pause:
            cenv.render()
            time.sleep(0.1)
    print()
    print("reward:",cr)

while 1:
    window_still_open = rollout(cenv)
    if window_still_open==False: break