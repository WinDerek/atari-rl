# Setup the environment

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

**Install dependencies**

```shell
$ conda install pip
$ python -m pip install gym==0.9.5 tensorflow==1.5.0 tensorpack==0.8.0 numpy==1.14.0 scipy==1.1.0 opencv-python==3.4.1.15
$ python -m pip install "gym[atari]"
```

**Download the model**

```shell
$ mkdir models
$ wget http://models.tensorpack.com/OpenAIGym/SpaceInvaders-v0.tfmodel -P models --no-check-certificate
```

## Legacy steps (PLEASE IGNORE THE FOLLOWING STEPS)

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
