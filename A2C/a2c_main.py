import sys
import os
# import wandb

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


import gym
from A2C.a2c_agent import A2Cagent
from envs.sample_env import PIDsampleEnv

def main():
    max_episode_num = 10 # set the max episode num as you want
    env = PIDsampleEnv(set_point=0.3)
    agent = A2Cagent(env)


    # train
    agent.train(max_episode_num, plot=True)

    # test
    agent.test(plot=True)

if __name__ == "__main__":
    main()