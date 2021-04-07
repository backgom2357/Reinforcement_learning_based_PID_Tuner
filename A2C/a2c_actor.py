import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Lambda
import tensorflow as tf

class ActorNetwork(Model):
    def __init__(self, action_dim, action_bound):
        super(ActorNetwork, self).__init__()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(16, activation='relu')
        self.out_mu = Dense(action_dim, activation='sigmoid')
        self.mu_adjust = Lambda(lambda x: x * action_bound)
        self.std_output = Dense(action_dim, activation='softplus')

    def call(self, i):
        output = self.d1(i)
        output = self.d2(output)
        output = self.d3(output)
        mu_output = self.out_mu(output)
        return self.mu_adjust(mu_output), self.std_output(output)

class Actor(object):
    """
        A2C Actor Neural Net
    """

    def __init__(self, state_dim, action_dim, action_bound, learning_rate):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        # set min and max of standard deviation
        self.std_bound = [1e-2, 1.0]

        # create actor neural net
        self.model = ActorNetwork(self.action_dim, self.action_bound)

        """
        tf 2.0
        """

        # loss and optimizer
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as g:
            mu_a, std_a = self.model(states)
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)
        dj_dtheta = g.gradient(loss, self.model.trainable_weights)
        grads = zip(dj_dtheta, self.model.trainable_weights)
        self.actor_optimizer.apply_gradients(grads)

    def model_initializer(self, state):
        self.states = state

    # log_policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    # get action
    def get_action(self, state):
        mu_a, std_a = self.model(np.reshape(state, [1, self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]

        std_a = tf.clip_by_value(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action

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

