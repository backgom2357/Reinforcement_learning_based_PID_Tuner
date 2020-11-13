import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from .ddpg_actor import Actor
from .ddpg_critic import Critic
from .replaybuffer import ReplayBuffer

class DDPGPIDTunner(object):

    def __init__(self, state_size, action_size, action_bound=100):
        
        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        # 상태와 행동의 크기
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_bound = action_bound

        ## 액터와 크리틱 생성
        self.actor = Actor(self.state_dim, self.action_dim,
                            self.action_bound, self.TAU, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.TAU, self.CRITIC_LEARNING_RATE)

        ## 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

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

    def save_model(self):
        self.actor.save_weights("./save_model/ddpg_actor_trained.h5")
        self.critic.save_weights("./save_model/ddpg_critic_trained.h5")