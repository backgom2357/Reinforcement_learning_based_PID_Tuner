import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
import tensorflow as tf

class Actor(object):

    """
    tensorflow 2.3
    DDPG Actor
    """

    def __init__(self, state_dim, action_dim, action_bound, tau, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau

        # Create Actor and Target Neural Network
        self.model, self.theta, self.states = self.build_network()
        self.target_model, self.target_theta, _ = self.build_network()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    # Actor Network
    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out = Dense(self.action_dim, activation='tanh')(h3)
        
        # Modify output
        action_output = Lambda(lambda x: x*self.action_bound)(out)

        model = Model(state_input, action_output)
        return model, model.trainable_weights, state_input
    
    # Predict
    def predict(self, state):
        return self.model.predict(np.reshape(state, [1, self.state_dim]))[0]
    
    # Predict from Target
    def target_predict(self, state):
        return self.target_model.predict(state)
    
    # Copy Actor Parameters to Target Parameters
    def update_target_network(self):
        theta, target_theta = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(theta)):
            target_theta[i] = self.tau * theta[i] + (1 - self.tau) * target_theta[i]
        self.target_model.set_weights(target_theta)
        
    # Train Actor
    def train(self, states, dq_das):
        with tf.GradientTape() as g:
            outputs = self.model(states)
        dj_dtheta = g.gradient(outputs, self.theta, -dq_das)
        grads = zip(dj_dtheta, self.theta)
        self.optimizer.apply_gradients(grads)
        
    # Save Actor Network
    def save_weights(self, path):
        self.model.save_weights(path)
        
    # Load Actor Network
    def load_weights(self, path):
        self.mdoel.load_weights(path + 'pendulum_actor.h5')