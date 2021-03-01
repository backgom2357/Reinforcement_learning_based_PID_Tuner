import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

def build_model(state_dim, action_dim):
    inputs = Input(shape=(state_dim))
    d1 = Dense(32, activation='relu')(inputs)
    d2 = Dense(64, activation='relu')(d1)
    d3 = Dense(64, activation='relu')(d2)
    output = Dense(action_dim)(d3)
    model = Model(inputs=inputs, outputs=output)
    return model