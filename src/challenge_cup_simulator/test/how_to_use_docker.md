# Docker 环境部署使用说明

NVIDIA显卡驱动、NVIDIA CUDA Toolkit还没安装的可以参考LeTools-Learning文档网站说明进行安装。

## 安装 Docker（仅在没有安装的情况下参考）

```bash
sudo apt update
sudo apt install git
sudo apt install docker.io
# 配置 NVIDIA Runtime
nvidia-ctk
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtime
# 输出中应包含 "nvidia" Runtime
```

## 使用 Docker 安装 ROS Noetic，仅供参考。

- 首先最好是加入一些源：

```bash
sudo vim /etc/docker/daemon.json
```

- 然后在这个json文件中写入一些镜像源：

```json
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.imgdb.de",
        "https://docker-0.unsee.tech",
        "https://docker.hlmirror.com",
        "https://docker.1ms.run",
        "https://func.ink",
        "https://lispy.org",
        "https://docker.xiaogenban1993.com"
    ]
}
```

- 然后保存文件并退出后，重启docker服务：

```shell
sudo systemctl daemon-reload && sudo systemctl restart docker
```

- 现在开始创建镜像，首先建立Dockerfile：
```shell
mkdir -p /path/to/save/docker/ros/image
cd /path/to/save/docker/ros/image
vim Dockerfile
```
然后在Dockerfile文件中写入如下内容：

```Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y locales tzdata gnupg lsb-release
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# 设置ROS的debian源
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加ROS的Keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 安装ROS Noetic
# 设置键盘布局为 Chinese
RUN apt-get update && \
    apt-get install -y keyboard-configuration apt-utils && \
    echo 'keyboard-configuration keyboard-configuration/layoutcode string cn' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/variant string ' | debconf-set-selections && \
    apt-get install -y ros-noetic-desktop-full && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

# 初始化rosdep
RUN rosdep init
```
写入完毕后保存退出。

- 执行ubuntu20.04 + ROS Noetic镜像的构建：

```shell
sudo docker build -t ubt2004_ros_noetic .
```

- 构建完成后进入镜像即可，初次启动容器加载镜像：

```shell
# GPU 启动（推荐）-v <宿主机路径>:<容器路径> 
sudo docker run -it --gpus all --network host -v /path/to/LeTools-Learning:/root/LeTools-Learning --shm-size=16g --runtime nvidia --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
```

之后每次加载：
```shell
sudo docker start ubuntu_ros_container
sudo docker exec -it ubuntu_ros_container /bin/bash
```

- 进入镜像后，初始化ros环境变量，然后启动roscore

```shell
source /opt/ros/noetic/setup.bash
roscore
```

无误的话，ubuntu20.04 + ros noetic的docker配置方式就结束了。

## Docker 容器内部署 Miniforge letools 环境

- 安装wget
```shell
apt update
apt install -y wget
```


- Miniforge安装

```shell
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh
chmod +x ./Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh #这里使用bash
source /root/.bashrc
```

- Conda配置：

```shell
conda create -n letools python=3.12
conda activate letools
```

- 安装依赖

```shell
cd /path/to/LeTools-Learning
source /opt/ros/noetic/setup.bash  # 进入python环境先source好ros自带的python库，建议这行写入/root/.bashrc
# Install dependencies
chmod +x setup_env.sh #脚本里面有GitHub拉代码的部分，自己带了gh-proxy加速
bash setup_env.sh    # 需确保 ROS Noetic 已安装 (推荐)
apt update
apt install ros-noetic-apriltag-ros
pip install pyarrow
```
- 后续衔接 LeTools-Learning文档网站 进行使用即可。