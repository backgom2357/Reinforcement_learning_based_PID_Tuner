# Reinforcement Learning based PID Tuner
This project is the implemetation of the Reinforcement Learning based Online PID Tuner. The Tuner is based on [A2C](https://arxiv.org/abs/1602.01783). I trained the RL tuner and tested on Lunarlander, one of OpenAi gym env..


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
    - state (5,) : Set Point, feedback, error, I-term, P
    - action (1,) : P
    - reward (1,) : if abs(error) in a certain range, give 1. Or, give -1


# Result

### Please check here - [Experiment Report (Korean)](https://www.notion.so/RL-based-PID-Tunner-add9501d8c8d422eba55101da27d9072)

<br>

## Pretrain result
- Before training

![101b](https://user-images.githubusercontent.com/30210944/113843222-5cbed700-97ce-11eb-8119-c382f48987ab.png)

- After training

![101a](https://user-images.githubusercontent.com/30210944/113843357-795b0f00-97ce-11eb-9060-2bc16a63cbf8.png)

- Training plot

![Untitled](https://user-images.githubusercontent.com/30210944/113843564-a4456300-97ce-11eb-9cac-64482840b10a.png)



## Test PID control with auto tuner in [Lunarlander-v2](https://gym.openai.com/envs/LunarLander-v2/)
It do not need any tuning process.
- Render

![tuner_applied](https://user-images.githubusercontent.com/30210944/113843894-f1c1d000-97ce-11eb-9b14-f00e9e22cfd9.gif)

- Error Plot

![image](https://user-images.githubusercontent.com/30210944/113844367-5ed56580-97cf-11eb-9cd3-8a7f8f6f42d0.png)

Orange line represents set-points, and blue line represents feedbacks. (left) Angular controller. (Right) Vertical controller.

# Usage
### Training
```
cd ./A2C/
python a2c_main.py
```
### Test
```
cd ./envs/
python ./LunarLanderContinuous_keyboard_agent_tuner_applied.py
```

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
