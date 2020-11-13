import sys
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import os

class A2CPIDTunner:
    def __init__(self, state_size, action_size, load_model=False):
        self.load_model = load_model
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.grad_bound = 0.
        
        self.std_bound = [1e-2, 1.0]

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()

        if self.load_model:
            self.actor.load_weights("./save_model/actor_trained.h5")
            self.critic.load_weights("./save_model/critic_trained.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        input_state = tf.keras.Input((self.state_size,))
        d1 = tf.keras.layers.Dense(24, activation='selu')(input_state)
        d2 = tf.keras.layers.Dense(24, activation='selu')(d1)
        out_mu = tf.keras.layers.Dense(self.action_size, activation='tanh')(d2) # tanh / linear
        out_std = tf.keras.layers.Dense(self.action_size, activation='softplus')(d2)
        actor =  tf.keras.Model(input_state, [out_mu, out_std])
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        input_state = tf.keras.Input((self.state_size,))
        d1 = tf.keras.layers.Dense(24, activation='relu')(input_state)
        d2 = tf.keras.layers.Dense(24, activation='relu')(d1)
        output = tf.keras.layers.Dense(1, activation='tanh')(d2)
        critic = tf.keras.Model(input_state, output)
        return critic

    # log_policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
    
    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, std = self.actor(np.reshape(state, [1, self.state_size]))
        mu = mu[0]
        std = std[0]

        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu, std, size=self.action_size)
        return action

    # 정책신경망을 업데이트하는 함수
    def train_actor(self, action, state, advantage):
        with tf.GradientTape() as t:
            mu_a, std_a = self.actor(state)
            log_policy_pdf = self.log_pdf(mu_a, std_a, action)
            loss = -K.sum(log_policy_pdf * advantage)
        g_theta = t.gradient(loss, self.actor.trainable_weights)
        grads = zip(g_theta, self.actor.trainable_weights)
#         grads = [(tf.clip_by_value(grad, -self.grad_bound, self.grad_bound), var) for grad, var in grads]
        self.actor_optimizer.apply_gradients(grads)

    # 가치신경망을 업데이트하는 함수
    def train_critic(self, state, target):
        with tf.GradientTape() as t:
            output = self.critic(state)
            loss = K.mean(K.square(target - output))
        g_omega = t.gradient(loss, self.critic.trainable_weights)
        grads = zip(g_omega, self.critic.trainable_weights)
#         grads = [(tf.clip_by_value(grad, -self.grad_bound, self.grad_bound), var) for grad, var in grads]
        self.critic_optimizer.apply_gradients(grads)

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        value = self.critic(state)[0]
        next_value = self.critic(next_state)[0]

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        advantage = reward - value + (1 - done)*(self.discount_factor * next_value)
        target = reward + (1 - done)*(self.discount_factor * next_value)
        
        self.train_actor(action, state, advantage)
        self.train_critic(state, target)
        
    def save_model(self):
        self.actor.save_weights("./save_model/actor_trained.h5")
        self.critic.save_weights("./save_model/critic_trained.h5")