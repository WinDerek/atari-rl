"""
Bot 3: A simple Q-learning agent for the FrozenLake environment.
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


# Hyperparameters
REPETITIONS_COUNT = 100
EPISODES_COUNT = 5000
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Total Average: %.2f (Episode %d)'


# episode_reward_2darray = np.zeros((REPETITIONS_COUNT, EPISODES_COUNT), dtype=float)


def print_report(reward_list, episode):
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


@ray.remote
def experiment(episodes_count):
    begin_time = time.time()

    num_episodes = episodes_count
    discount_factor = 0.8
    learning_rate = 0.9
    report_interval = 500

    # 创建环境
    env = gym.make('FrozenLake-v0')

    # 创建一个初始值全零的 Q 表格
    # Q 表格用一个 numpy 二维数组表示
    # 行数为状态空间的大小（即环境的观测空间的大小），列数为动作空间的大小
    q_2darray = np.zeros((env.observation_space.n, env.action_space.n))

    episode_reward_array = np.zeros(num_episodes, dtype=float)

    # 训练，每一个循环为一个 episode
    for episode in range(1, num_episodes + 1):
        # 每个 episode 刚开始的时候，重置环境
        state = env.reset()

        episode_reward = 0.0

        # 循环直至一个 episode 结束
        # 每一轮循环 agent 会执行一次 action，即一个 step
        while True:
            # 给动作加入噪声来增加 agent 的探索程度
            noise = np.random.random((1, env.action_space.n)) / (episode**2.)
            action = np.argmax(q_2darray[state, :] + noise)

            # 让 agent 在环境中执行动作
            # state_to: 执行之后 agent 所处的状态
            # reward: agent 获得的立即回报
            # done: 表示当前 episode 是否结束的指示变量（bool）
            state_to, reward, done, _ = env.step(action)
            
            # 更新 Q 表格
            q_target = reward + discount_factor * np.max(q_2darray[state_to, :])
            q_2darray[state, action] = (1 - learning_rate) * q_2darray[state, action] + learning_rate * q_target
            
            episode_reward += reward
            state = state_to
            
            # 如果当前 episode 已经结束
            if done:
                episode_reward_array[episode - 1] = episode_reward

                if episode % report_interval == 0:
                    print_report(episode_reward_array, episode)
                
                break
    print_report(episode_reward_array, -1)


    end_time = time.time()
    print('Experiment completed in {:s}.'.format(format_time(end_time - begin_time)))

    return episode_reward_array


def main():
    # for i in range(REPETITIONS_COUNT):
    #     print("Running experiment {:d}:".format(i + 1))
    #     run_experiment(i)

    futures = []
    for repetition_index in range(REPETITIONS_COUNT):
        futures.append(experiment.remote(EPISODES_COUNT))

    # Merge the results
    episode_reward_array_list = ray.get(futures)
    episode_reward_2darray = np.stack(episode_reward_array_list)
    print("episode_reward_2darray.shape:", episode_reward_2darray.shape)

    # Persist the results
    results_path = Path("./results/q_learning")
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
