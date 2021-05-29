"""
Q network agent.
"""

from typing import List
import gym
import numpy as np
import random
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# 超参数
NUM_EPISODES = 4000
discount_factor = 0.99
learning_rate = 0.15
report_interval = 500
# 探索概率。启发式的想法，episode 越大（训练得越久），探索概率越小。
exploration_probability = lambda episode: 50. / (episode + 10)
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'


def one_hot(i: int, n: int) -> np.array:
    """Implements one-hot encoding by selecting the ith standard basis vector"""
    return np.identity(n)[i].reshape((1, -1))


def print_report(rewards: List, episode: int):
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


def main():
    # 创建环境
    env = gym.make('FrozenLake-v0')
    
    # 回报列表
    reward_list = []

    # 初始化
    n_obs, n_actions = env.observation_space.n, env.action_space.n
    obs_t_ph = tf.placeholder(shape=[1, n_obs], dtype=tf.float32)
    obs_tp1_ph = tf.placeholder(shape=[1, n_obs], dtype=tf.float32)
    action_ph = tf.placeholder(tf.int32, shape=())
    rew_ph = tf.placeholder(shape=(), dtype=tf.float32)
    q_target_ph = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)

    # 创建计算图
    W = tf.Variable(tf.random_uniform([n_obs, n_actions], 0, 0.01))
    q_current = tf.matmul(obs_t_ph, W)
    # print("type(obs_t_ph):", type(obs_t_ph))
    # print("obs_t_ph.shape:", obs_t_ph.shape)
    q_target = tf.matmul(obs_tp1_ph, W)

    q_target_max = tf.reduce_max(q_target_ph, axis=1)
    q_target_sa = rew_ph + discount_factor * q_target_max
    q_current_sa = q_current[0, action_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    pred_action_ph = tf.argmax(q_current, 1)

    # 创建梯度下降（gradient descent）优化器用于训练
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    update_model = trainer.minimize(error)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # 训练循环，每一次迭代就是一个 episode
        for episode in range(1, NUM_EPISODES + 1):
            # 每个 episode 刚开始的时候，重置环境
            obs_t = env.reset()

            # 循环直至一个 episode 结束
            # 每一轮循环 agent 会执行一次 action，即一个 step
            episode_reward = 0
            while True:
                # 选取最优的 action，或者均匀地随机选一个 action
                obs_t_oh = one_hot(obs_t, n_obs)
                action = session.run(pred_action_ph, feed_dict={obs_t_ph: obs_t_oh})[0]
                # 以一定的概率进行探索，随机性来自于 np.random.rand() 函数
                if (np.random.rand(1) < exploration_probability(episode)): # X ~ Unif[0, 1], P(X < 0.1) = 0.1
                    action = env.action_space.sample()
                obs_tp1, reward, done, _ = env.step(action)

                # 训练模型
                obs_tp1_oh = one_hot(obs_tp1, n_obs)
                q_target_val = session.run(q_target, feed_dict={obs_tp1_ph: obs_tp1_oh})
                session.run(update_model, feed_dict={
                    obs_t_ph: obs_t_oh,
                    rew_ph: reward,
                    q_target_ph: q_target_val,
                    action_ph: action
                })

                # 记录数据
                episode_reward += reward
                obs_t = obs_tp1

                # 如果当前 episode 已经结束
                if done:
                    reward_list.append(episode_reward)

                    if (episode % report_interval == 0):
                        print_report(reward_list, episode)
                    
                    break
        print_report(reward_list, -1)


if __name__ == '__main__':
    main()
