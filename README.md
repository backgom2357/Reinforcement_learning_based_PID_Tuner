# Reinforcement Learning based PID Tuner
The implemetation of the Reinforcement Learning based PID Tuner. Try to build tuner system with [PPO](https://arxiv.org/abs/1707.06347) and [Double DQN](https://arxiv.org/abs/1509.06461) algorithm. Combine RL algorithm to PID contorller. Using two different method to update (P, I, D) tuple.


# Procedure

## Flowchart
![RL based PID Tunner](https://user-images.githubusercontent.com/30210944/109543644-e6adbd00-7b09-11eb-8ff3-16863d9db2e7.png)

## Pseudo code
```
Init (P,I,D) of the environment
Init the policy π
for episode = 0, M do
	Inint state
	Set done = False
	Reset the environment
	while not done do
		action = π(state)
		next_state, reward, done = step(action)
		Train π
		state = next_state
	end while
end for
```

## Environment
Using Simple PID control example to build PID environment.
- MDP
    - state (50,) : The stack of feedbacks for 50 steps
    - action (3,) : Values to update (P, I, D) 
    - reward (1,) : (max(abs(error sum for 50 steps), -160) + 80)/80 (10 if done properly)
    - done (1,) : abs(error sum for 50 steps < 0.05 * (# of steps)

- Method
    - reset() : 
    <br>Initialize variables (including (P, I, D))
    - step(action) : 
    <br> Run 50 steps of PID control example with given (P, I, D).
    <br> Return next_state, reward, done, error_sum
        - (method 1) (P, I, D) = (P + action[0], I + action[1], D + action[2])
        - (method 2) (P, I, D) = (action[0], action[1], action[2])
    - plot() :
    <br>Plot PID control result

# Result

### Please check here - [Experiment Report (Korean)](https://www.notion.so/RL-based-PID-Tunner-add9501d8c8d422eba55101da27d9072)

<br>

## (P, I, D) from Tuner
method 1
- (0.8311611133067119, 1.067838020900425, -0.0006975578421864661)
- (0.8799401544613751, 1.1651741645523848, 0.002376775442951846)

method 2
- (0.816297709601598, 1.1406315660539015, 0.002037705470730431)
- (1.253136190601821, 0.620026107056804, -0.002720052821954503)

## Apply PID control in [Lunarlander-v2](https://gym.openai.com/envs/LunarLander-v2/)
- Given Control

![result](https://user-images.githubusercontent.com/30210944/109546631-d13a9200-7b0d-11eb-9202-d2d6cb590191.gif)

- PID Control <br> (P, I, D) = (1.253136190601821, 0.620026107056804, -0.002720052821954503)

![pid_ctr_result](https://user-images.githubusercontent.com/30210944/109546804-00e99a00-7b0e-11eb-84b0-f61d8cc3db40.gif)


# Usage
### Training
```
cd ./PPO/
python ppo_main.py
```
### Test
Follow [pid_control_test.ipynb](https://github.com/backgom2357/reinforcement-learning-based-PID-tunner/blob/master/pid_control_test.ipynb)

# requirements
```
tensorflow==2.2.0
scikit-learn==0.23.2
matplotlib==3.3.3
gym
```

# reference
https://github.com/ivmech/ivPID

https://github.com/pasus/Reinforcement-Learning-Book
