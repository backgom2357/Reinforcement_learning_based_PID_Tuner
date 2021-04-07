from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

class CriticNetwork(Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(16, activation='relu')
        self.v_output = Dense(1, activation='linear')

    def call(self, inputs):
        output = self.d1(inputs)
        output = self.d2(output)
        output = self.d3(output)
        return self.v_output(output)

class Critic(object):

    """
        A2C Critic Neural Net
    """

    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # create critic neural net
        self.model = CriticNetwork()

        # set training method
        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')

    # update neural net with batch data
    def train_on_batch(self, states, td_targets):
        return self.model.train_on_batch(states, td_targets)

    # save Critic parameter
    def save_weights(self, path):
        self.model.save_weights(path)

    # load Critic parameter
    def load_weights(self, path):
        self.model.load_weights(path)
