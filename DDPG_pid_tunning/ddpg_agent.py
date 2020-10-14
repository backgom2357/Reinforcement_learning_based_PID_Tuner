import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from ddpg_actor import Actor
from ddpg_critic import Critic
from replaybuffer import ReplayBuffer

class DDPGagent(object):

    def __init__(self, env):
        
        # Hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        try:
            self.action_dim = env.action_space.shape[0]
            # get action bound
            self.action_bound = env.action_space.high[0]
        except:
            self.action_dim = env.action_space.n
            self.action_bound = 1

        ## create actor and critic networks
        self.actor = Actor(self.state_dim, self.action_dim,
                            self.action_bound, self.TAU, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.TAU, self.CRITIC_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []

    ## Ornstein Uhlenbeck Noise
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)

    ## computing TD target: y_k = r_k + gamma*Q(s_k+1, a_k+1)
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.actor.update_target_network()
        self.critic.update_target_network()

        for ep in range(int(max_episode_num)):
            # reset OU noise
            pre_noise = np.zeros(self.action_dim)
            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()
            while not done:
                # visualize the environment
                self.env.render()
                # pick an action: shape = (1,)
                action = self.actor.predict(state)
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                # clip continuous action to be within action_bound
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                # observe reward, new_state
                next_state, reward, done, _ = self.env.step(np.argmax(action))
                # add transition to replay buffer
                train_reward = (reward + 8) / 8
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_size > 1000:  # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    # predict target Q-values
                    target_qs = self.critic.target_predict([next_states, self.actor.target_predict(next_states)])
                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs, dones)
                    # train critic using sampled batch
                    self.critic.train_on_batch(states, actions, y_i)
                    # Q gradient wrt current policy
                    s_actions = self.actor.model.predict(states) # shape=(batch, 1),
                    # caution: NOT self.actor.predict !
                    # self.actor.model.predict(state) -> shape=(1,1)
                    # self.actor.predict(state) -> shape=(1,) -> type of gym action
                    s_grads = self.critic.dq_da(states, s_actions)
                    dq_das = np.array(s_grads).reshape((-1, self.action_dim))
                    # train actor
                    self.actor.train(states, dq_das)
                    # update both target network
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                # update current state
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            #print('Now save')
        #     self.actor.save_weights("/home/backgom2357/workplace/math_RL/DDPG_pid_tunning/save_weights/pendulum_actor.h5")
        #     self.critic.save_weights("/home/backgom2357/workplace/math_RL/DDPG_pid_tunning/save_weights/pendulum_critic.h5")

        # np.savetxt('/home/backgom2357/workplace/math_RL/DDPG_pid_tunning/save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()