from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
import tensorflow as tf

class Critic(object):
    
    """
    tensorflow 2.3
    DDPG Actor
    """
    
    def __init__(self, state_dim, action_dim, tau, learning_rate):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate
        
        # Build Critic and Target Network
        self.model, self.states, self.actions = self.build_network()
        self.target_model, _, _ = self.build_network()
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        
    def build_network(self):
        state_input = Input((self.state_dim,))
        action_input = Input((self.action_dim,))
        x1 = Dense(64, activation='relu')(state_input)
        x2 = Dense(32, activation='linear')(x1)
        a1 = Dense(32, activation='relu')(action_input)
        a2 = Dense(32, activation='linear')(a1)
        h2 = concatenate([x2, a2], axis=-1)
        h3 = Dense(16, activation='relu')(h2)
        q_output = Dense(1, activation='tanh')(h3)
        model = Model([state_input, action_input], q_output)
        return model, state_input, action_input
    
    # Trarget Critic Network Predict
    def target_predict(self, inp):
        return self.target_model.predict(inp)
        
    # Copy Critic Parameters to Target Parameters
    def update_target_network(self):
        phi = self.model.get_weights()
        target_phi = self.target_model.get_weights()
        for i in range(len(phi)):
            target_phi[i] = self.tau * phi[i] + (1 - self.tau) * target_phi[i]
        self.target_model.set_weights(target_phi)
        
    # Calculate dQ/da
    def dq_da(self, states, actions):
        with tf.GradientTape() as g:
            actions = tf.convert_to_tensor(actions)
            g.watch(actions)
            outputs = self.model([states, actions])
        q_grads = g.gradient(outputs, actions)
        return q_grads
    
    # Update Network with Batch Data
    def train_on_batch(self, states, actions, td_targets):
        self.model.train_on_batch([states, actions], td_targets)
        
    # Save Critic Network
    def save_weights(self, path):
        self.model.save_weights(path)
        
    # Load Critic Network
    def load_weights(self, path):
        self.mdoel.load_weights(path + 'pendulum_critic.h5')