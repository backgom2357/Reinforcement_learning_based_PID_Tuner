import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

from A2C.a2c_actor import Actor
from A2C.a2c_critic import Critic

class A2Cagent(object):

    def __init__(self, env):

        # hyperparameter
        self.GAMMA = 0.95
        self.BATCH_SIZE = 8
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space
        # action dimension
        self.action_dim = env.action_space
        # max action size
        self.action_bound = env.action_bound

        # create Actor and Critic neural nets
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        # total reward of a episode
        self.save_epi_reward = []

    # calculate advantages and TD targets
    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = reward
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        return advantage, y_k

    # extract data from batch
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack

    # train agent
    def train(self, max_episode_num, plot=False, on_wandb = False):

        if on_wandb:
            import wandb
            wandb.init(project='pidTuner', entity='diominor')
            
        best_episode_reward = -1

        # repeat for each episode
        for ep in range(int(max_episode_num)):

            # init batch
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [] ,[]
            # init episode
            time, episode_reward, done = 0, 0, False
            # reset env and observe initial state
            state = self.env.reset()

            mean_v = 0

            while not done:

                # visualize env
                # self.env.render()

                # get action
                action = self.actor.get_action(state)

                # bound action range
                action = np.clip(action, -self.action_bound, self.action_bound)

                # observe next state, reward
                next_state, reward, done, error = self.env.step(action)

                # reshape
                state = np.reshape(state, [1, self.state_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])

                # calculate state value
                v_value = self.critic.model(state)
                next_v_value = self.critic.model(next_state)

                # calculate advantage and TD target
                advantage, y_i = self.advantage_td_target(reward, v_value, next_v_value, done)

                # append to batch
                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)

                print(f'step : {time}, pid : {self.env.Kp:+.3f}, {self.env.Ki:+.3f}, {self.env.Kd:+.3f}, reward : {reward[0][0]:+.3f}, error : {error:3.1f}', end='\r')
                if plot==True:
                    self.env.plot(done)

                # wait for full batch
                if len(batch_state) < self.BATCH_SIZE:

                    # update state
                    state = next_state[0]
                    episode_reward += reward[0][0]
                    time += 1
                    continue

                # train
                # extract from batch
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                advantages = self.unpack_batch(batch_advantage)

                # clear batch
                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                # critic neural net update
                self.critic.train_on_batch(states, td_targets)

                # actor neural net update
                self.actor.train(states, actions, advantages)

                 # update state
                state = next_state[0]
                episode_reward += reward[0][0]
                time += 1

                mean_v += np.mean(self.critic.model(np.array([state])))

            print()
            print(f'epi : {ep}, episode_reward : {episode_reward:+.3f}, mean v : {mean_v/time}')
            if on_wandb: wandb.log({'episode reward': episode_reward, 'mean v': mean_v/time})
            self.save_epi_reward.append(episode_reward)
            
            if best_episode_reward < episode_reward:
                self.actor.save_weights(os.path.join(CURRENT_DIR, f'save_weights/tunner_actor_crt.h5'))
                self.critic.save_weights(os.path.join(CURRENT_DIR, f'save_weights/tunner_critic_crt.h5'))

            if (ep+1)%10==0:
                self.actor.save_weights(os.path.join(CURRENT_DIR, f'save_weights/tunner_actor{ep}.h5'))
                self.critic.save_weights(os.path.join(CURRENT_DIR, f'save_weights/tunner_critic{ep}.h5'))


    def test(self, plot):

        # Initialize model
        state = np.zeros((1, self.state_dim))
        self.actor.get_action(state)
        self.critic.model(state)
        self.actor.load_weights(os.path.join(CURRENT_DIR, "save_weights/tunner_actor_crt.h5"))
        self.critic.load_weights(os.path.join(CURRENT_DIR, "save_weights/tunner_critic_crt.h5"))
        
        # init episode
        time, episode_reward, done = 0, 0, False
        # reset env and observe initial state
        state = self.env.reset()

        while not done:

            # visualize env
            # self.env.render()

            # get action
            action = self.actor.get_action(state)

            # bound action range
            action = np.clip(action, -self.action_bound, self.action_bound)

            # observe next state, reward
            next_state, reward, done, error = self.env.step(action)

            # reshape
            state = np.reshape(state, [1, self.state_dim])
            next_state = np.reshape(next_state, [1, self.state_dim])
            action = np.reshape(action, [1, self.action_dim])
            reward = np.reshape(reward, [1, 1])

            # update state
            state = next_state[0]
            episode_reward += reward[0]
            time += 1

            print(f'step : {time}, pid : {self.env.Kp:+.3f}, {self.env.Ki:+.3f}, {self.env.Kd:+.3f}, reward : {reward[0][0]:+.3f}, error : {error:3.1f}', end='\r')
            if plot==True:
                self.env.plot(done)

        


    # graph episodes and rewards
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()


















