# 使用docker与docker-compose例子

为了让用户更便捷启动项目，我们准备了docker与docker-compose方式快速上手项目，采用docker-compose是为了方便用户在物理机编写项目调试项目，又能迅速映射物理机文件到docker容器内部路径，实现物理机调试，docker容器内运行，解决环境依赖于配置的问题。

## 一、环境配置
需要先安装docker与docker-compose（本案例以ubuntu为例）

```bash
$ sudo apt update
$ sudo apt install -y ca-certificates curl gnupg lsb-release

启用官方存储库:
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
使用apt命令安装docker
$ sudo apt-get update
$ sudo apt install docker-ce docker-ce-cli containerd.io -y
docker 包安装完成后，请将本地用户加入 docker 组，以便该用户无需 sudo 即可执行 docker 命令

$ sudo usermod -aG docker $USER
$ newgrp docker

```
通过执行以下命令验证 Docker 版本
```bash
$ docker version
```
![1.jpg](../media/docker-version.jpg)

安装docker-compose

```bash
pip install docker-compose
```

查看docker-compose是否安装完毕。

![2.jpg](../media/docker-compose-version.png)

## 二、启动容器

在启动容器之前，先修改一下docker-compose.yml文件里面的volumes路径

”volumes："里面冒号左侧为物理机项目路径，右侧为映射入容器的项目路径，这样做的好处是调试本地代码的时候不用频繁切换入容器，且能实时同步效果到容器中。

修改红框内路径改为本地物理机项目路径

![docker-compose-localpath](../media/docker-compose-localpath.jpg)

下一步我们切换到docker目录下执行：

Docker-compose up

该命令会自动匹配目录下的docker-compose.yml文件，并执行，即可看到自动拉取docker镜像（如果pull太慢，可以尝试换docker镜像国内地址，根据系统不同请自行查阅切换docker镜像地址方法）

![pulling images](../media/docker-compose-pulling-images.jpg)

下载完成后会自动运行容器，此时另外新开一个窗口执行以下命令可查看容器运行状态：

```
docker ps
```

![docker-ps](../media/docker-ps.jpg)

此时执行exce命令进入容器：

```
docker exec -it {容器的ID} /bin/bash
```

![docker-exec](../media/docker-exec.jpg)

进入容器后就能切换到/home/的默认目录下（上面已经在docker-compose.yml文件指定了映射路径）

然后容器内执行conda命令（我们尽量保留容器内的conda 环境让用户可以在本容器里继续创建别的conda环境）

```
conda activate chatglmeft
```

![docker-conda](../media/docker-conda.jpg)

最后用户就能自由的在容器内部执行python web_demo.py等命令启动界面并在物理机访问调试、测试了。



