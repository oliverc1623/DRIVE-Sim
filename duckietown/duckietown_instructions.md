# Duckietown Nautilus Instruction

Follow these steps to get a headless version of Duckietown working on the Nautilus server via a jupyter notebook.

1. Log in to the Nautilus Jupyterhub

https://jupyterhub-west.nrp-nautilus.io/hub/home

Check the mount that includes PyTorch

2. Create a conda environment with Python 3.9

```
conda create --name [env name] python=3.9
```

3. Activate Conda by calling Source

```
source /opt/conda/bin/activate [env name]
```

4. Upgrade pip

```
pip install --upgrade pip
```

5. cd into gym-duckietown

If you don't have gym-duckietown, git clone https://github.com/duckietown/gym-duckietown.git

Install duckietown package

```
pip3 install -e .
```

6. If you get a build error

```
pip install pyglet==1.5.11
```

7. Install xvfb

```
sudo apt-get update
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y ffmpeg
sudo apt install xvfb
```

Run to list xvfb processes:

```
ps aux | grep Xvfb
```

8. Start xvfb process


If you want to run Xvfb in the background, you can use the & symbol at the end of the command:
```
Xvfb :1 -screen 0 1280x1024x24 &
```

Check if Xvfb :1 was created 
```
ps aux | grep Xvfb
```

Remember that if you're running this on a server or headless environment, you might also need to set the DISPLAY environment variable to point to the Xvfb display. For example:
```
export DISPLAY=:1
```

## How to setup a kernel for Notebooks
```
source activate [myenv]
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

9. Open Jupyter notebook

Make sure to set your display environment:
```
import os

os.environ["DISPLAY"] = ":1"  # the display number is the virtual display number used with Xvfb
```

10. Import gym-duckietown

```
from gym_duckietown.envs import DuckietownEnv
```
