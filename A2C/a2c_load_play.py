import gym
from A2C.a2c_agent import A2Cagent

def main():
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = A2Cagent(env)

    agent.test()

    state = env.reset()
    time = 0


    while True:
        env.render()
        action = agent.actor.predict(state)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()