from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

"""
    TF2.2 functional API
"""

class Critic(object):

    """
        PPO Critic Neural Net
    """

    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # create critic neural net
        self.model = self.build_network()

        # set training method
        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')

    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        v_output = Dense(1, activation='linear')(h3)

        model = Model(state_input, v_output)
        return model

    # update neural net with batch data
    def train_on_batch(self, states, td_targets):
        return self.model.train_on_batch(states, td_targets)

    # save Critic parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Critic parameter
    def load_weights(self, path):
        self.model.load_weights(path+'pendulum_critic.h5')
