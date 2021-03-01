import numpy as np

class ReplayMemory(object):
    def __init__(self, replay_memory_size, state_dim):
        self.rm_size = replay_memory_size

        # init state, action, reward, next_state, done
        self.seqs = np.zeros((replay_memory_size, state_dim), dtype=np.uint32)
        self.rewards = np.zeros(replay_memory_size, np.float32)
        self.rewards[-1] = 777
        self.actions = np.zeros(replay_memory_size, np.uint8)
        self.next_seqs = np.zeros((replay_memory_size, state_dim), dtype=np.uint32)
        self.dones = np.zeros(replay_memory_size, np.bool)

        self.crt_idx = 0

    def is_full(self):
        return self.rewards[-1] != 777

    def append(self, seq, action, reward, next_seq, done):
        self.seqs[self.crt_idx] = seq
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_seqs[self.crt_idx] = next_seq
        self.dones[self.crt_idx] = done

        self.crt_idx = (self.crt_idx + 1) % self.rm_size

    def sample(self, batch_size):
        rd_idx = np.random.choice((1 - self.is_full())*self.crt_idx+self.is_full()*self.rm_size, batch_size)
        batch_seqs = self.seqs[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_seqs = self.next_seqs[rd_idx]
        batch_dones = self.dones[rd_idx]

        return batch_seqs, batch_actions, batch_rewards, batch_next_seqs, batch_dones