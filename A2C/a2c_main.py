import sys

import wandb
sys.path.append('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/')

import gym
from A2C.a2c_agent import A2Cagent
from envs.sample_env import PIDsampleEnv

def main():
    max_episode_num = 100000
    env = PIDsampleEnv(set_point=0.3)
    agent = A2Cagent(env)


    # train
    # agent.train(max_episode_num, plot=True)

    # test
    agent.test(plot=True)

if __name__ == "__main__":
    main()