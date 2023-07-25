# VISTA Nautilus Instruction

Follow these steps to get a headless version of VISTA working on the Nautilus server via a jupyter notebook.

1. Log in to the Nautilus Jupyterhub

https://jupyterhub-west.nrp-nautilus.io/hub/home

Check the mount that includes PyTorch

2. Create a conda environment with Python 3.8

```
conda create --name [env name] python=3.8
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

If you want to run Xvfb in the background, you can use the & symbol at the end of the command:
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

4. Install torch 

```
pip3 install torch torchvision
```

5. Install the following python dependencies

Refer to the original VISTA's [yaml](https://github.com/vista-simulator/vista/blob/main/environment.yaml) for specific versions.

- opencv-python (pip install opencv-python-headless)
- ffio
- shapely
- descartes
- matplotlib
- pyrender
- pickle5
- h5py

6. Install VISTA package

```
pip install vista
```

7. Begin each python file with 

```
import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
```

Replace ":1" if you chose another virtual display port. 

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