from network import build_model
from replay_memory import ReplayMemory
from utils import preprocess, normalize
import numpy as np
import tensorflow as tf
# import wandb

class DDQNTunner:
    def __init__(self, config, env):
        
        # Get  Config
        self.cf = config

        # environment
        self.env = env
        # state dimension
        self.state_dim = env.observation_space
        # action dimension
        self.action_dim = env.action_space
        # max action size
        self.action_bound = env.action_bound + 1

        # Setting Replay Memory
        self.rm = ReplayMemory(self.cf.REPLAY_MEMORY_SIZE, self.state_dim)
        
        # Build Model
        self.q = build_model(self.state_dim, self.action_dim)
        self.target_q = build_model(self.state_dim, self.action_dim)

        # Optimizer and Loss for Training
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.cf.LEARNING_RATE, clipnorm=10.)
        self.loss = tf.keras.losses.Huber()
        self.q.summary()

        # Save Logs
        # wandb.init(
        #     project="fully_conv_layer_test",
        #     name='vanilla_DQN_'+ str(env)[20:-3],
        #     config=self.cf.WANDB)

    def get_action(self, state):
        """
        Epsilon Greedy
        """
        q = self.q(state)[0]
        return (np.argmax(q), q) if self.cf.epsilon < np.random.rand() else (np.random.randint(self.action_dim), q)

    def model_train(self):
        # Sample From Replay Memory
        states, actions, rewards, next_states, dones = self.rm.sample(self.cf.BATCH_SIZE)

        # Epsilon Decay (+ exponentially)
        if self.cf.epsilon > self.cf.FINAL_EXPLORATION:
            self.cf.epsilon -= (1 + self.cf.FINAL_EXPLORATION)/(self.cf.FINAL_EXPLORATION_FRAME*self.cf.epsilon)
        
        # Update Weights
        with tf.GradientTape() as g:
            # Action from current q function
            current_actions = np.argmax(self.q(next_states), axis=1)

            # q value with next state and action from current q function
            next_q_from_target = self.target_q(next_states)
            next_q_from_target_with_action = tf.reduce_sum(next_q_from_target * tf.one_hot(current_actions, self.action_dim), axis=1)

            # Calculate Targets
            targets = rewards + (1 - dones) * (self.cf.DISCOUNT_FACTOR * next_q_from_target_with_action)

            predicts = self.q(states)
            predicts = tf.reduce_sum(predicts * tf.one_hot(actions, self.action_dim), axis=1)
            loss = self.loss(targets, predicts)

        g_theta = g.gradient(loss, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(g_theta, self.q.trainable_weights))

    def run(self, max_episode):
        
        # For the Logs
        sum_mean_q, episodic_rewards = 0, 0

        # Initalizing
        episode = 0
        frames, action = 0, 0
        best_pid = (0,0,0)
        max_error = 1e10000
        state = self.env.reset()

        while episode < max_episode:

            # Interact with Environmnet
            (action, q) = self.get_action(np.expand_dims(state, axis=0))
            env_action = [1 if i==action else 0 for i in range(3)]
            next_state, reward, done, error = self.env.step(env_action)

            if error < max_error:
                best_pid = (self.env.Kp,self.env.Ki,self.env.Kd)
                max_error = error

            # Append To Replay Memeory
            self.rm.append(state, action, reward, next_state, done)
            print(f'step : {frames}, pid : {self.env.Kp:+.3f}, {self.env.Ki:+.3f}, {self.env.Kd:+.3f}, reward : {reward:+.3f}, error : {error:3.1}', end='\r')

            frames += 1

            # Start Training After Collecting Enough Samples
            if self.rm.crt_idx < self.cf.REPLAY_START_SIZE and not self.rm.is_full():
                state = next_state
                continue
            
            # Training
            self.model_train()
            state = next_state

            episodic_rewards += reward
            sum_mean_q += np.mean(q)

            # Update Target Q
            if frames % self.cf.TARGET_NETWORK_UPDATE_FREQUENCY  == 0:
                self.target_q.set_weights(self.q.get_weights())
            

            if done:
                episodic_mean_q = sum_mean_q/frames
                episode += 1

                # Update Logs
                print(f'epi : {episode}, pid : {best_pid[0]:+.3f}, {best_pid[1]:+.3f}, {best_pid[2]:+.3f}, episode_reward : {episodic_rewards:+.4f} episodic_mean_q : {episodic_mean_q:+.4f}')
                # wandb.log()

                # Save Model
                        
                # Initializing
                sum_mean_q, episodic_rewards = 0, 0
                frames, action = 0, 0
                best_pid = (0,0,0)
                max_error = 1e10000
                initial_state = self.env.reset()
                state = initial_state