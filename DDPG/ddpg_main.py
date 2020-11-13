# DDPG main

import gym
from ddpg_agent import DDPGagent

def main():

    GAMES = {0:"Pendulum-v0", 1:'CartPole-v0'}
    
    max_episode_num = 2000
    env = gym.make(GAMES[1])
    agent = DDPGagent(env)

    agent.train(max_episode_num)

    agent.plot_result()


if __name__=="__main__":
    main()