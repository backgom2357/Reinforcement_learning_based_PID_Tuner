import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PID import PID
import time

class PIDsampleEnv(PID):
    
    def __init__(self, P=0.0, I=0.0, D=0.0, set_point=1):
        super().__init__()
        
        self.Kp = P
        self.Ki = I
        self.Kd = D
        
        # self.alpha = alpha        # mutiply to p,i,d
        self.set_point = set_point
        self.end = 50

        self.count = 0

        self.observation_space = (8)
        self.action_space = (3)
        self.action_bound = (1)

        self.feedback = 0
        self.timestep = 0
        self.SetPoint = 0.0

        self.reward = 0
        self.done = False

        # plot
        self.time_list = []
        self.feedbacks = []
        self.setPoints = []
        
    def reset(self):
        super().clear()
        # self.Kp = np.random.uniform(0,2)
        # self.Ki = np.random.uniform(0,2)
        # self.Kd = np.random.uniform(0,0.01)
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.
        self.count = 0

        self.feedback = 0
        self.timestep = 0
        self.SetPoint = 0.0
        
        self.reward = 0
        self.done = False
        # plot
        self.time_list = []
        self.feedbacks = []
        self.setPoints = []
        return np.array([self.feedback, self.last_error, self.PTerm, self.ITerm, self.DTerm, self.Kp, self.Ki, self.Kd])
    
    def step(self, action):

        self.timestep += 1

        self.Kp = action[0]
        self.Ki = action[1]
        self.Kd = 0.01 * action[2]

        next_state = []
        
        self.update(self.feedback)
        output = self.output
            
        self.feedback += output

        if self.timestep > 10:
            self.SetPoint = self.set_point
            time.sleep(0.01)
        
        next_state.append(self.SetPoint)
        next_state.append(self.last_error/(abs(self.SetPoint)*1.5) if self.SetPoint else 0)
        next_state.extend([self.PTerm, self.ITerm, self.DTerm])
        next_state.extend([self.Kp, self.Ki, self.Kd])
        
        # reward = 1 if abs(self.last_error) < 0.01 else -1
        reward = -abs(self.last_error)
        
        self.count += 1

        if self.count > 30:
            self.done = True
        if abs(self.last_error) > abs(self.SetPoint)*1.5:
            reward = -5
            self.done = True
        
        if self.timestep <= 9:
            reward = 0

        # plot
        self.time_list.append(self.timestep)
        self.feedbacks.append(self.feedback)
        self.setPoints.append(self.SetPoint)
        
        return next_state, reward, self.done, self.last_error
    
    def plot(self):

        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()

        plt.figure(2)
        plt.clf()
        plt.plot(self.time_list, self.feedbacks)
        plt.plot(self.time_list, self.setPoints)
        plt.xlabel('time (s)')
        plt.ylabel('PID (PV)')
        plt.title(f'{self.timestep}step')
        # if epi:
        #     plt.savefig(f"./{epi}.png")
        plt.pause(0.01)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

        plt.ioff()