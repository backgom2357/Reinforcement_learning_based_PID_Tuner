import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ppo_actor import Actor
from ppo_critic import Critic

class PPOagent(object):

    def __init__(self, env):

        # hyperparameter
        self.GAMMA = 0.95
        self.GAE_LAMBDA = 0.9
        self.BATCH_SIZE = 64
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 10
        self.t_MAX = 4

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space.shape[0]
        # action dimension
        self.action_dim = env.action_space.shape[0]
        # max action size
        self.action_bound = env.action_space.high[0]

        # create Actor and Critic neural nets
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE, self.RATIO_CLIPPING)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        # total reward of a episode
        self.save_epi_reward = []

    # calculate GAE and TD targets
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_stop_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_stop_targets[k] = gae[k] + v_values[k]
        return gae, n_stop_targets

    # train agent
    def train(self, max_episode_num):

        # repeat for each episode
        for ep in range(int(max_episode_num)):

            # init states, actions, reward
            states, actions, rewards = [], [], []
            # init log old policy pdf
            log_old_policy_pdfs = []
            # init episode
            time, episode_reward, done = 0, 0, False
            # reset env and observe initial state
            state = self.env.reset()

            while not done:

                # visualize env
                self.env.render()

                # get action
                mu_old, std_old, action = self.actor.get_policy_action(state)

                # bound action range
                action = np.clip(action, -self.action_bound, self.action_bound)

                # calculate log old policy pdf
                var_old = std_old**2
                log_old_policy_pdf = -0.5 * (action - mu_old)**2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                # observe next state, reward
                next_state, reward, done, _ = self.env.step(action)

                # save to batch
                states.append(state)
                actions.append(action)
                rewards.append((reward + 8) / 8)  # modify reward range
                log_old_policy_pdfs.append(log_old_policy_pdf)

                # state update
                state = next_state
                episode_reward += reward
                time += 1

                if len(states) == self.t_MAX or done:

                    # get data from batch
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    log_old_policy_pdfs = np.array(log_old_policy_pdfs)


                    # calculate n-step TD target and advantage
                    next_state = np.reshape(next_state, [1, self.state_dim])
                    next_v_value = self.critic.model(next_state)
                    v_values = self.critic.model(states)
                    gaes, y_i = self.gae_target(rewards, v_values, next_v_value, done)

                    for _ in range(self.EPOCHS):
                        # train actor network
                        self.actor.train(states, actions, gaes, log_old_policy_pdfs)
                        # train critic network
                        self.critic.train_on_batch(states, y_i)

                    # clear batch
                    states, actions, rewards = [], [], []
                    log_old_policy_pdfs = []

            print("Epi: ", ep+1, "Time: ", time, "Reward: ", episode_reward)
            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_actor.h5")

        np.savetxt('.save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    # graph episodes and rewards
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


















