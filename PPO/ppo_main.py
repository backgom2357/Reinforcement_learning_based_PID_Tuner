import sys

import wandb
sys.path.append('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/')

from envs.sample_env import PIDsampleEnv
from ppo_agent import PPOTunner

def main():
    max_episode_num = 100000
    env = PIDsampleEnv(set_point=1)
    tunner = PPOTunner(env)
    tunner.train(max_episode_num, plot=1, on_wandb=True)

if __name__ =='__main__':
    main()