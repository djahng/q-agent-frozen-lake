import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent


def main():
    avg_scores = []
    scores = []
    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)

    agent = Agent(n_states=16, n_actions=4, learn_rate=0.001, gamma=0.9,
                  epsilon=1.0, epsilon_min=0.01, epsilon_decrement=0.9999995)

    for i in range(1000000):
        score = 0
        terminated = False
        observation, info = env.reset()

        while not terminated:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)

            agent.learn(observation, action, reward, observation_)

            observation = observation_
            score += reward

        scores.append(score)

        if i % 100 == 0:
            avg = np.array(scores[-100:]).mean()
            avg_scores.append(avg)

            if i % 1000 == 0:
                print(f"Episode {i}, Win Percentage {avg:0.2f}, Epsilon {agent.epsilon:0.2f}")

    plt.plot(avg_scores)
    plt.show()


if __name__ == '__main__':
    main()
