# Atari RL

To-Do List:
- [ ] Migrate TensorFlow 1 to Tensorflow 2.
- [ ] Typeset a LaTeX documentation.

**Requirements**

**Bias-Variance for Deep Reinforcement Learning: How To Build a Bot for Atari with OpenAI Gym**

References:

- [https://github.com/alvinwan/bots-for-atari-games](https://github.com/alvinwan/bots-for-atari-games)

## 1. Setup the Environment

**Install Miniconda**

For Windows (x86_64), download the [installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) to install and you can skip the rest of this section.

For macOS and Linux, execute the following shell commands to install Miniconda:

```shell
$ curl -OJ https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash ./Miniconda3-latest-Linux-x86_64.sh
```

> - References: https://docs.conda.io/en/latest/index.html
> - As the official link is unreachable or too slow, here I replace the original link with the Tsinghua mirror.

**Create a new conda environment**

```shell
$ conda create --name atari_env python=3.6
$ conda activate atari_env
```

**New instructions**

```shell
$ conda install pip
$ python -m pip install gym==0.9.5 tensorflow==1.5.0 tensorpack==0.8.0 numpy==1.14.0 scipy==1.1.0 opencv-python==3.4.1.15
$ python -m pip install "gym[atari]"
```

```shell
$ mkdir models
$ wget http://models.tensorpack.com/OpenAIGym/SpaceInvaders-v0.tfmodel -P models --no-check-certificate
```

**Install OpenCV (4.2.0)**

```shell
$ conda install -c conda-forge opencv=4.2.0
```

**Install pip**

```shell
$ conda install pip
```

**Install OpenAI Gym**

For non-Windows systems:

```shell
$ pip install gym
$ pip install "gym[atari]"
```

For Windows systems:

```shell
$ pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
$ pip install gym
$ pip install "gym[atari]"
```

References:
- [https://github.com/openai/atari-py/issues/69#issuecomment-554135051](https://github.com/openai/atari-py/issues/69#issuecomment-554135051)
- [https://blog.csdn.net/ScienceVip/article/details/105097833](https://blog.csdn.net/ScienceVip/article/details/105097833)

**Install Tensorflow, tensorpack and numpy**

```shell
$ pip install tensorflow
$ pip install tensorpack
$ pip install numpy
```

## 2. Baseline Random Agent

Source code: `random_agent.py`

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

Source code: `q_learning_agent.py`

## 5. Deep Q-Learning Agent for FrozenLake

Source code: `q_network_agent.py`

## 6. Bias-Variance Tradeoffs

## 7. Least Squares Agent for FrozenLake

Source code: `least_squares_agent.py`

## 8. Deep Q-Learning Agent for Space Invaders

Source code: `dqn_agent.py`
