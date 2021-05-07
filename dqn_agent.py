"""
Bot 6 - A full-featured deep Q-learning agent.
"""

import argparse
import cv2
import gym
import numpy as np
import random
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from a3c import a3c_model
# random.seed(0)  # make results reproducible
# tf.set_random_seed(0)

num_episodes = 10


def downsample(state):
    """将输入的状态中的帧缩放到 84x84 的大小。
    """
    return cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)[None]


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run DQN agent.')
    parser.add_argument('--model', type=str, help='Path to model', default='models/SpaceInvaders-v0.tfmodel')
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()

    # 创建环境
    env = gym.make('SpaceInvaders-v0')
    
    # 回报列表
    reward_list = []

    # 构建用于训练的模型，此处是一个 A3C 模型
    model = a3c_model(load=args.model)

    # 训练循环，每一次迭代就是一个 episode
    for _ in range(num_episodes):
        # 每个 episode 刚开始的时候，重置环境和相关数据
        episode_reward = 0
        state_list = [downsample(env.reset())]

        # 循环直至一个 episode 结束
        # 每一轮循环 agent 会执行一次 action，即一个 step
        while True:
            if len(state_list) < 4:
                action = env.action_space.sample()
            else:
                # 沿着第三维（RGB）进行合并，相当于是将过去最新的四帧沿着 RGB 的那一维合并。
                frames = np.concatenate(state_list[-4:], axis=3)
                # print("frames.shape:", frames.shape) # (1, 84, 84, 12)
                action = np.argmax(model([frames]))

            # 如果需要 GUI 可视化
            if args.visual:
                env.render()
            
            # 让 agent 在环境中执行动作
            # state: 执行之后 agent 所处的状态。对于该环境 state 的类型是 numpy.ndarray，形状为 (210, 160, 3)，第三维为 RGB。
            # reward: agent 获得的立即回报
            # done: 表示当前 episode 是否结束的指示变量（bool）
            state, reward, done, _ = env.step(action)
            # print("type(state):", type(state)) # numpy.ndarray
            # print(state.shape) # (210, 160, 3)

            # 记录数据
            state_list.append(downsample(state))
            # print("type(state_list[-1]):", type(state_list[-1])) # numpy.ndarray
            # print(state_list[-1].shape) # (1, 84, 84, 3)
            episode_reward += reward
            
            # 如果当前 episode 已经结束
            if done:
                print('Reward: %d' % episode_reward)
                reward_list.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(reward_list) / len(reward_list)))


if __name__ == '__main__':
    main()
