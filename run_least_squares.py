"""
Least squares Q-learning agent for FrozenLake.
"""


from pathlib import Path
import pickle
import random
import time

import gym
import numpy as np
import ray

from util.time_utils import format_time


total_begin_time = time.time()


# Initialize ray
ray.init(dashboard_host="0.0.0.0")


# 超参数
REPETITIONS_COUNT = 100
num_episodes = 5000
discount_factor = 0.85
learning_rate = 0.9
# w 的学习率
w_lr = 0.5
# 每隔 500 个 episode 输出一次数据
report_interval = 500

report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'


def makeQ(model):
    """Returns a Q-function, which takes state -> distribution over actions"""
    return lambda X: X.dot(model)


def initialize(shape):
    """Initialize model"""
    W = np.random.normal(0.0, 0.1, shape)
    Q = makeQ(W)
    return W, Q


def train(X, y, W):
    """Train the model, using solution to ridge regression"""
    I = np.eye(X.shape[1])
    newW = np.linalg.inv(X.T.dot(X) + 10e-4 * I).dot(X.T.dot(y))
    W = w_lr * newW + (1 - w_lr) * W
    Q = makeQ(W)
    return W, Q


def one_hot(i, n):
    """Implements one-hot encoding by selecting the ith standard basis vector"""
    return np.identity(n)[i]


def print_report(rewards, episode):
    """Print rewards report for current episode
    - Average for last 100 episodes
    - Best 100-episode average across all time
    - Average for all episodes across time
    """
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode))


@ray.remote
def experiment():
    begin_time = time.time()

    # Create the environment
    env = gym.make('FrozenLake-v0')
    
    episode_reward_array = np.zeros(num_episodes, dtype=float)

    n_obs, n_actions = env.observation_space.n, env.action_space.n
    W, Q = initialize((n_obs, n_actions))
    states, labels = [], []
    for episode in range(1, num_episodes + 1):
        if len(states) >= 10000:
            states, labels = [], []
        state = one_hot(env.reset(), n_obs)
        episode_reward = 0
        while True:
            states.append(state)
            noise = np.random.random((1, n_actions)) / episode
            action = np.argmax(Q(state) + noise)
            state2, reward, done, _ = env.step(action)

            state2 = one_hot(state2, n_obs)
            Qtarget = reward + discount_factor * np.max(Q(state2))
            label = Q(state)
            label[action] = (1 - learning_rate) * label[action] + learning_rate * Qtarget
            labels.append(label)

            episode_reward += reward
            state = state2
            if len(states) % 10 == 0:
                W, Q = train(np.array(states), np.array(labels), W)
            if done:
                episode_reward_array[episode - 1] = episode_reward

                if episode % report_interval == 0:
                    print_report(episode_reward_array, episode)
                break
    print_report(episode_reward_array, -1)

    end_time = time.time()
    print("Experiment completed in {:s}.".format(format_time(end_time - begin_time)))

    return episode_reward_array


def main():
    futures = []
    for repetition_index in range(REPETITIONS_COUNT):
        futures.append(experiment.remote())

    # Merge the results
    episode_reward_array_list = ray.get(futures)
    episode_reward_2darray = np.stack(episode_reward_array_list)
    print("episode_reward_2darray.shape:", episode_reward_2darray.shape)

    # Persist the results
    results_path = Path("./results/least_squares")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path.joinpath("results.pkl")
    results = { "episode_reward_2darray": episode_reward_2darray }
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
        print("The results have been persisted into the file \"{:s}\".".format(str(results_file)))
    
    total_end_time = time.time()
    print("Total time usage: {:s}.".format(format_time(total_end_time - total_begin_time)))

if __name__ == '__main__':
    main()
