"""
Bot 2 -- Make a random, baseline agent for the SpaceInvaders game.
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
    
    rewards = []

    for episode_index in range(NUM_EPISODES):
        env.reset()
        episode_reward = 0

        while True:
            # Conduct random action
            _, reward, done, _ = env.step(env.action_space.sample())
            
            episode_reward += reward
            
            if done:
                print('Episode #{:d}, reward: {:.2f}'.format(episode_index + 1, episode_reward))
                rewards.append(episode_reward)
                break
    
    print('Average reward over {:d} episodes: {:.2f}'.format(NUM_EPISODES, sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()
