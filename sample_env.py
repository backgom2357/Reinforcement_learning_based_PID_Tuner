import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PID import PID

class PIDsampleEnv(PID):
    
    def __init__(self, P=0.2, I=0.0, D=0.0, alpha=0.01, set_point=1):
        super().__init__()
        
        self.Kp = P
        self.Ki = I
        self.Kd = D
        
        self.alpha = alpha        # mutiply to p,i,d
        self.set_point = set_point
        self.end = 50

        self.count = 0
        
    def reset(self):
        self.clear()
        self.count = 0
        return np.zeros(self.end)
    
    def step(self, action):
        
        error_sum = 0
        feedback = 0
        reward  = 0
        done = False
        self.SetPoint = 0.0

        self.Kp += self.alpha * action[0]
        self.Ki += self.alpha * action[1]
        self.Kd += self.alpha * 0.1 * action[2]
        next_state = []
        
        for i in range(1, self.end+1):
            self.update(feedback)
            output = self.output
            
            if self.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                self.SetPoint = 1

            error_sum += abs(self.last_error)
            
            next_state.append(feedback)
        
        reward = max(-error_sum + self.SetPoint, -1)
        
        if error_sum < 0.04 * (self.end - 11):
            reward = 10
            done = True
        
        self.count += 1

        if self.count > 100:
            done = True
        
        return next_state, reward, done, error_sum
    
    def plot(self, epi=None):

        feedback = 0.0
        self.SetPoint = 0.0

        feedback_list = []
        time_list = []
        setpoint_list = []

        for i in range(1, self.end+1):
            
            self.update(feedback)
            output = self.output
            
            setpoint_list.append(self.SetPoint)

            if self.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                self.SetPoint = 1
            
            feedback_list.append(feedback)    
            time_list.append(i)
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()

        plt.figure(2)
        plt.clf()
        plt.plot(time_list, feedback_list)
        plt.plot(time_list, setpoint_list)
        plt.xlim((0, self.end))
        plt.xlabel('time (s)')
        plt.ylabel('PID (PV)')
        plt.title('TEST PID')
        plt.ylim((-self.set_point*1.3, self.set_point*1.3))
        # if epi:
        #     plt.savefig(f"./{epi}.png")
        plt.pause(0.01)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

        plt.ioff()