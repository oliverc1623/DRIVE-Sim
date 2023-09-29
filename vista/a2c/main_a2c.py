import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('VistaEnv.py'))))
from VistaEnv import VistaEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('CustomCNN.py'))))
from CustomCNN import CustomCNN
import copy

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import torch

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

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
trace_root = "../vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center", 
    "20210726-155941_lexus_devens_center_reverse", 
    "20210726-184624_lexus_devens_center", 
    "20210726-184956_lexus_devens_center_reverse", 
]
trace_paths = [os.path.join(trace_root, p) for p in trace_path]

display_config = dict(road_buffer_size=1000, )

preprocess_config = {"crop_roi": True}

env = VistaEnv(trace_paths = trace_paths, 
               trace_config = trace_config,
               car_config = car_config,
               display_config = display_config,
               preprocess_config = preprocess_config,
               sensors_configs = [camera_config])

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)

# check_env(env, warn=True)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=2)
timesteps = 50_000
model.learn(total_timesteps=timesteps, progress_bar=True)

# Save the agent
model.save("vista_a2c_mycnn_50000_400x640")
