# Atari RL

To-Do List:
- [ ] Migrate TensorFlow 1 to Tensorflow 2.
- [ ] Typeset a LaTeX documentation.

**Requirements**

**Bias-Variance for Deep Reinforcement Learning: How To Build a Bot for Atari with OpenAI Gym**

References:

- [https://github.com/alvinwan/bots-for-atari-games](https://github.com/alvinwan/bots-for-atari-games)

## 1. Setup the Environment

[setup.md](setup.md)

## 2. Baseline Random Agent

`python random_agent.py`

## 3. Reinforcement Learning Basics

### 3.1 Dynamic Programming

**Policy evaluation**

<img src="./figure/policy_evaluation_discrete_case.png" width="560">

**Policy improvement**

<img src="./figure/policy_improvement_discrete_case.png" width="700">

**Policy iteration**

<img src="./figure/policy_iteration_diagram.png" width="200">

**Value iteration**

<img src="./figure/value_iteration_discrete_case.png" width="700">

## 4. Q-Learning Agent for FrozenLake

**Q-learning**

<img src="./figure/q_learning_discrete_case.png" width="700">

### 4.1 运行单次

```shell
$ python q_learning_agent.py
```

### 4.2 运行多次取平均并画图

```shell
$ python run_q_learning.py
$ python plot.py q_learning
```

Episode reward of Q-learning agent on FrozenLake-v0 (averaged over 100 repetitions)

<img src="./results/q_learning/figures/q_learning.png" width="600">

> 说明：浅色（original）的每个点的横坐标是 episode，纵坐标是 episode reward，并且这是重复一百次独立实验取平均的结果。深色（smooth）是对浅色曲线每隔 100 个数据点算一次平均值而（也就是每　100 个 episode 算一次平均值）得到的更平滑的曲线。其他的图都是类似的。

## 5. Deep Q-Learning Agent for FrozenLake

### 5.1 运行单次

```shell
$ python q_network_agent.py
```

### 5.2 运行多次取平均并画图

```shell
$ python run_q_network.py
$ python plot.py q_network
```

Episode reward of Q network agent on FrozenLake-v0 (averaged over 100 repetitions)

<img src="./results/q_network/figures/q_network.png" width="600">

## 6. Bias-Variance Tradeoffs

## 7. Least Squares Agent for FrozenLake

`python least_squares_agent.py`

## 8. Deep Q-Learning Agent for Space Invaders

- Without GUI: `python dqn_agent.py`
- With GUI: `python dqn_agent.py --visual`
