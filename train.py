import numpy as np
import matplotlib.pyplot as plt
from sample_env import PIDsampleEnv
from A2C.a2c_agent import A2CPIDTunner
from DDPG.ddpg_agent import DDPGPIDTunner

class Train:

    def __init__(self, env, state_size, action_size, agent, max_epi):
        self.env = env
        self.agent = agent
        self.state_size = state_size
        self.action_size = action_size
        self.max_epi = max_epi

        self.error_list = []
        self.done_list = []
        self.least_error = 10
        self.least_pid = (0,0,0)

        self.done_tinue = 0

    def train(self, is_DDGP=False):

        if is_DDGP:
                self.agent.actor.update_target_network()
                self.agent.critic.update_target_network()
        
        pre_noise = np.zeros(self.action_size)

        min_error = 999999
        
        for e in range(self.max_epi):

            done = False
            noise = self.agent.ou_noise(pre_noise, dim=self.action_size)
            state = self.env.reset()
            # state = np.reshape(state, [1, state_size])
            action = 0
            score = 0

            while not done:
                if is_DDGP:
                    action = self.agent.actor.predict(state)
                    action = np.clip(action + noise, -self.agent.action_bound, self.agent.action_bound)
                else:
                    action = self.agent.get_action(state)

                next_state, reward, done, info = self.env.step(action)
                reward = max(-100, reward)
                if reward < min_error:
                    min_error = reward
                    reward += 10
                # next_state = np.reshape(next_state, [1, state_size])

                self.agent.buffer.add_buffer(state, action, reward, next_state, done)
                
                if is_DDGP and self.agent.buffer.buffer_count() > 1000:
                    # print(train)
                    # raise Exception
                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.agent.buffer.sample_batch(self.agent.BATCH_SIZE)
                    # predict target Q-values
                    target_qs = self.agent.critic.target_predict([next_states, self.agent.actor.target_predict(next_states)])
                    # compute TD targets
                    y_i = self.agent.td_target(rewards, target_qs, dones)
                    # train critic using sampled batch
                    self.agent.critic.train_on_batch(states, actions, y_i)
                    # Q gradient wrt current policy
                    s_actions = self.agent.actor.model.predict(states) # shape=(batch, 1),
                    # caution: NOT self.actor.predict !
                    # self.actor.model.predict(state) -> shape=(1,1)
                    # self.actor.predict(state) -> shape=(1,) -> type of gym action
                    s_grads = self.agent.critic.dq_da(states, s_actions)
                    dq_das = np.array(s_grads).reshape((-1, self.action_size))
                    # train actor
                    self.agent.actor.train(states, dq_das)
                    # update both target network
                    self.agent.actor.update_target_network()
                    self.agent.critic.update_target_network()
                # else:
                #     self.agent.train_model(state, action, reward, next_state, done)
                
                state = next_state
                pre_noise = noise
                score += reward
                
                self.error_list.append(info)
                
                if done:
            #         print(f'done!')
                    self.done_list.append((self.env.Kp, self.env.Ki, self.env.Kd))
            #         env.plot(e)
                    if self.env.last_error < self.least_error:
                        self.least_error = self.env.last_error
                        self.least_pid = (self.env.Kp, self.env.Ki, self.env.Kd)
                        self.done_tinue += 1
                        if self.done_tinue > 20:
                            self.agent.save_model()
                            break
                    else:
                        self.done_tinue = 0
                    self.env.plot()
                    print('epi : {:6.0f}, pid : {:7.4f}, {:7.4f}, {:7.4f}, score : {:4.0f}, error : {:10.3}, d-count : {}'.format(e, 
                                                                                                                                self.env.Kp, 
                                                                                                                                self.env.Ki, 
                                                                                                                                self.env.Kd, 
                                                                                                                                score, 
                                                                                                                                info,
                                                                                                                                len(self.done_list)), end='\r', flush=True)

                    

if __name__ == "__main__":

    env = PIDsampleEnv(alpha=0.01, set_point=1)

    state_size = 50
    action_size = 3

    # 액터-크리틱(A2C) 에이전트 생성
    agent = DDPGPIDTunner(state_size, action_size, action_bound=1)

    train = Train(env, state_size, action_size, agent, 100)
    train.train(is_DDGP=True)
    train.env.plot()