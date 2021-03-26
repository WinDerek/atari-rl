"""
Bot 2: A random, baseline agent for the SpaceInvaders environment.
"""


import gym
import random


# Make results reproducible
random.seed(0)


NUM_EPISODES = 10


def main():
    # Create the environment
    env = gym.make('SpaceInvaders-v0')

    # Make results reproducible
    env.seed(0)
    
    reward_list = []

    for episode_index in range(NUM_EPISODES):
        env.reset()
        episode_reward = 0

        while True:
            # Conduct random action
            random_action = env.action_space.sample()
            _, reward, done, _ = env.step(random_action)
            
            episode_reward += reward
            
            if done:
                print('Episode #{:d}, reward: {:.2f}'.format(episode_index + 1, episode_reward))
                reward_list.append(episode_reward)
                break
    
    print('Average reward over {:d} episodes: {:.2f}'.format(NUM_EPISODES, sum(reward_list) / len(reward_list)))


if __name__ == '__main__':
    main()
