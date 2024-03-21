#!/usr/bin/env python
# coding: utf-8

# In[1]:


# local imports
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('VistaEnv.py'))))
from VistaEnv import VistaEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('CustomCNN.py'))))
from CustomCNN import CustomCNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('SeqTransformer.py'))))
from SeqTransformer import SeqTransformer
from VistaMAEnv import VistaMAEnv

# Standard Torch
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Initialize the simulator
trace_config = dict(
    road_width=4,
    reset_mode='default',
    master_sensor='camera_front',
)
car_config = dict(
    length=5.,
    width=2.,
    wheel_base=2.78,
    steering_ratio=14.7,
    lookahead_road=True,
)
camera_config = {'type': 'camera',
                 'name': 'camera_front',
                 'rig_path': './RIG.xml',
                 'optical_flow_root': '../data_prep/Super-SloMo/slowmo',
                 'size': (400, 640)}
ego_car_config = copy.deepcopy(car_config)
ego_car_config['lookahead_road'] = True
trace_root = "vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center", 
    "20210726-155941_lexus_devens_center_reverse", 
    "20210726-184624_lexus_devens_center", 
    "20210726-184956_lexus_devens_center_reverse", 
]
trace_paths = [os.path.join(trace_root, p) for p in trace_path]
display_config = dict(road_buffer_size=1000, )
preprocess_config = {
    "crop_roi": True,
    "resize": True,
    "grayscale": True,
    "binary": False
}
env = VistaEnv(trace_paths = trace_paths, 
       trace_config = trace_config,
       car_config = car_config,
       display_config = display_config,
       preprocess_config = preprocess_config,
       sensors_configs = [camera_config])


# In[3]:


ob, info = env.reset()
plt.imshow(ob[0], cmap='gray')


# In[4]:


print(ob.shape)


# # Obstacle avoidance

# In[5]:


# Initialize the simulator
trace_config = dict(
    road_width=4,
    reset_mode='default',
    master_sensor='camera_front',
)
car_config = dict(
    length=5.,
    width=2.,
    wheel_base=2.78,
    steering_ratio=14.7,
    lookahead_road=True,
)
sensors_config = [
    dict(
        type='camera',
        # camera params
        name='camera_front',
        size=(400, 640), # for lighter cnn 
        # rendering params
        use_lighting=False,
    )
]
task_config = dict(n_agents=2,
                    mesh_dir="carpack01",
                    init_dist_range=[10., 20.],
                    init_lat_noise_range=[0., 0.])
display_config = dict(road_buffer_size=1000, )
preprocess_config = {
    "crop_roi": True,
    "resize": True,
    "grayscale": True,
    "binary": False,
    "seq": False,
}
ego_car_config = copy.deepcopy(car_config)
ego_car_config['lookahead_road'] = True
trace_root = "vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center", 
    "20210726-155941_lexus_devens_center_reverse", 
    "20210726-184624_lexus_devens_center", 
    "20210726-184956_lexus_devens_center_reverse", 
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]


# In[3]:


env = VistaMAEnv(
    trace_paths=trace_path,
    trace_config=trace_config,
    car_configs=[car_config] * task_config['n_agents'],
    sensors_configs=[sensors_config] + [[]] *
    (task_config['n_agents'] - 1),
    preprocess_config=preprocess_config,
    task_config=task_config
)


# In[4]:


ob, info = env.reset()
print(ob.shape)


# In[6]:


plt.imshow(ob[0], cmap='gray')


# In[ ]:




