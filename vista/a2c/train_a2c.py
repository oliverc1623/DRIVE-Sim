import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('VistaEnv.py'))))
from VistaEnv import VistaEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('CustomCNN.py'))))
from CustomCNN import CustomCNN
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('SeqTransformer.py'))))
from SeqTransformer import SeqTransformer
from typing import Callable

# Standard Torch
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SB3
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecVideoRecorder, VecFrameStack, StackedObservations
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

device = ("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

def make_env(rank: int, seed: int = 47):
    """
    Utility function for multiprocessed env.

    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
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
        env.set_seed(seed + rank)
        env.reset()
        return env
    set_random_seed(seed + rank)
    time.sleep(1)
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return p

learning_configs = {
    "policy_type": CustomCNN,
    "total_timesteps": 500_000,
    "env_id": "VISTA",
    "learning_rate": 0.0001, #linear_schedule(0.0001),
    "n_steps": 2048,
    "batch_size": 32,
    "ent_coef": 0.0,
}

if __name__ == "__main__":
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=25, verbose=1)
    torch.cuda.empty_cache()
    num_cpu = 8 
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # Create log dir
    log_dir = f"/mnt/persistent/lane-follow-a2c/tmp_{sys.argv[1]}/"
    os.makedirs(log_dir, exist_ok=True)
    vec_env = VecMonitor(vec_env, log_dir, ('out_of_lane', 'exceed_max_rot', 'agent_done', 'course_completion_rate'))
    policy_kwargs = dict(
        features_extractor_class=learning_configs['policy_type'],
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = A2C(
        "CnnPolicy", 
        vec_env,
        learning_rate=learning_configs['learning_rate'],
        n_steps=learning_configs['n_steps'],
        verbose=1,
        device=device
    )
    timesteps = learning_configs['total_timesteps']
    model.learn(
        total_timesteps=timesteps, 
        callback=callback_max_episodes,
        progress_bar=True
    )
    # Save the agent
    model.save(f"/mnt/persistent/lane-follow-a2c/tmp_{sys.argv[1]}/a2c-model-trial{sys.argv[1]}_customCNN")
