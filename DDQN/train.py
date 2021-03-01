import sys
sys.path.append('/home/diominor/Workspace/reinforcement-learning-based-PID-tunner/')

from envs.sample_env import PIDsampleEnv
from DDQN.agent import DDQNTunner
from DDQN.config import Config

def main(cf):
    
    max_episode = 1000
    env = PIDsampleEnv(alpha=0.01)
    ag = DDQNTunner(cf, env)
    ag.run(max_episode)

if __name__ == "__main__":
    cf = Config()
    main(cf)