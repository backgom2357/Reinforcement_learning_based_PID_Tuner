import gym
from ppo_agent import PPOagent

def main():
    max_episode_num = 1000
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = PPOagent(env)

    # train
    agent.train(max_episode_num)

    # visualization
    agent.plot_result()

if __name__ =='__main__':
    main()