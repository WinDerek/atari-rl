"""
Bot 3: A simple Q-learning agent for the FrozenLake environment.
"""


from typing import List
import numpy as np
import random

import gym


# Make results reproducible
random.seed(0)
np.random.seed(0)


# Hyperparameters
num_episodes = 4000
discount_factor = 0.8
learning_rate = 0.9
report_interval = 500
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f (Episode %d)'


def print_report(reward_list: List, episode: int):
    """Prints reward list report for current episode
    - Average for last 100 episodes
    - Best 100-episode average across all time
    - Average for all episodes across time
    """
    print(report % (
        np.mean(reward_list[-100:]),
        max([np.mean(reward_list[i:i+100]) for i in range(len(reward_list) - 100)]),
        np.mean(reward_list),
        episode))


def main():
    # Create the environment
    env = gym.make('FrozenLake-v0')

    # Make results reproducible
    env.seed(0)

    reward_list = []

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        while True:
            noise = np.random.random((1, env.action_space.n)) / (episode**2.)
            action = np.argmax(Q[state, :] + noise)
            state2, reward, done, _ = env.step(action)
            Qtarget = reward + discount_factor * np.max(Q[state2, :])
            Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * Qtarget
            episode_reward += reward
            state = state2
            if done:
                reward_list.append(episode_reward)

                if episode % report_interval == 0:
                    print_report(reward_list, episode)
                
                break
    print_report(reward_list, -1)


if __name__ == '__main__':
    main()
