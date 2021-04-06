import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Lambda, Input
import tensorflow as tf

"""
    TF2.2 functional API
"""

class Actor(object):
    """
        PPO Actor Neural Net
    """

    def __init__(self, state_dim, action_dim, action_bound, learning_rate, ratio_clipping):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.ratio_clipping = ratio_clipping

        # set min and max of standard deviation
        self.std_bound = [1e-3, 1]

        # create actor neural net
        self.model, self.theta = self.build_network()

        """
        tf 2.0
        """

        # loss and optimizer
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out_mu = Dense(self.action_dim, activation='relu')(h3)
        std_output = Dense(self.action_dim, activation='softplus')(h3)

        # bound mean
        mu_output = Lambda(lambda x: x*self.action_bound)(out_mu)
        model = Model(state_input, [mu_output, std_output])
        return model, model.trainable_weights

    def train(self, states, actions, advantages, log_old_policy_pdf):
        with tf.GradientTape() as g:
            mu_a, std_a = self.model(states)
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

            # ratio of two policy & target function surrogate
            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.ratio_clipping, 1.0 + self.ratio_clipping)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)
        dj_dtheta = g.gradient(loss, self.theta)
        grads = zip(dj_dtheta, self.theta)
        self.actor_optimizer.apply_gradients(grads)

    # log_policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    # get action
    def get_policy_action(self, state):
        mu_a, std_a = self.model(np.reshape(state, [1, self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]

        std_a = tf.clip_by_value(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return mu_a, std_a, action

    # calculate mean
    def predict(self, state):
        mu_a, _ = self.model(np.reshape(state, [1, self.state_dim]))
        return mu_a[0]

    # save Actor parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Actor parameter
    def load_weights(self, path):
        self.model.load_weights(path)

