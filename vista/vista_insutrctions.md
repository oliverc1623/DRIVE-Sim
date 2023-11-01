# VISTA Nautilus Instruction

Follow these steps to get a headless version of VISTA working on the Nautilus server via a jupyter notebook.

### 1. Log in to the Nautilus Jupyterhub

https://jupyterhub-west.nrp-nautilus.io/hub/home

Check the mount that includes PyTorch

### 2. Create a conda environment with Python 3.8

```
conda create --name [env name] python=3.8
```

### 3. Install system dependencies

```
sudo apt-get update
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y ffmpeg
sudo apt install xvfb
```

##### 3.1. Start xvfb process

If you want to run Xvfb in the background, you can use the & symbol at the end of the command:
Be sure to run with sudo
```
Xvfb :1 -screen 0 1280x1024x24 &
```

Remember that if you're running this on a server or headless environment, you might also need to set the DISPLAY environment variable to point to the Xvfb display. For example:
```
export DISPLAY=:1
```

Run to list xvfb processes:

```
ps aux | grep Xvfb
```

### 7. Begin each python file with 

```
import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
```
Replace ":1" if you chose another virtual display port. 

### 8. Install these packages

```
pip install stable-baselines3[extra]
pip install swig
pip install gymnasium[box2d]
pip install vista
pip install shapely
pip install ffio
pip install pyrender
pip install descartes
pip install wandb
pip install vit_pytorch
```

## Alternative VISTA Nautilus installation

1. Create a conda environment with Python 3.8

```
conda create --name [env name] python=3.8
```

2. Install all dependencies from my own yaml file

```
conda env update -n [my_env] --file environment.yaml
```

3. Install system dependencies

```
sudo apt-get update
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y ffmpeg
sudo apt-get install -y xvfb python-opengl x11-utils &> /dev/null
```

3.1. Start xvfb process


## How to setup a kernel for Notebooks
```
source activate [myenv]
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

# Stable Baselines 3 Install Instructions

Assuming X is installed on your server run:

```
pip install stable-baselines3[extra]
```

Run the two lines below to install Box2D

```
pip install swig
pip install gymnasium[box2d]
```

Run `pip install gymnasium[mujoco]` if you're using pybullet. 
