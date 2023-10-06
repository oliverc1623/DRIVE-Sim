import gymnasium as gym
from gymnasium import spaces

from typing import Optional, List, Dict, Any
import numpy as np
import random 
from skimage.transform import resize

from vista import World
from vista import Display
from vista.utils import logging, misc


def default_terminal_condition(task, agent_id, **kwargs):
    """ An example definition of terminal condition. """

    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]

    def _check_out_of_lane():
        road_half_width = agent.trace.road_width / 2. # 4 for training, 2 for eval
        return np.abs(agent.relative_state.x) > road_half_width

    def _check_exceed_max_rot():
        maximal_rotation = np.pi / 10.
        return np.abs(agent.relative_state.yaw) > maximal_rotation

    out_of_lane = _check_out_of_lane()
    exceed_max_rot = _check_exceed_max_rot()
    agent_done = agent.done
    done = out_of_lane or exceed_max_rot or agent_done
    other_info = {
        'done': done,
        'out_of_lane': out_of_lane,
        'exceed_max_rot': exceed_max_rot,
        'agent_done': agent_done
    }

    return done, other_info


def default_reward_fn(task, agent_id, **kwargs):
    """ An example definition of reward function. """
    reward = 0 if kwargs['done'] else 1

    return reward, {}


def get_rotation_penalty(agent_orientation, target_orientation, max_rotation_penalty):
    orientation_difference = abs(agent_orientation - target_orientation)
    rotation_threshold = 0.01  # Adjust this threshold as needed
    rotation_penalty = max(0, orientation_difference - rotation_threshold)
    penalty = -rotation_penalty * max_rotation_penalty
    return penalty

def lane_reward_fn(task, agent_id, prev_yaw, **kwargs):
    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]
    
    road_width = agent.trace.road_width
    z_lat = road_width / 2
    q_lat = np.abs(agent.relative_state.x)
    lane_reward = 1 - (q_lat/z_lat)**2

    rotation_penalty = get_rotation_penalty(prev_yaw, agent.ego_dynamics.numpy()[2], 1)
    # print(f"rotation penalty: {rotation_penalty}")

    reward = lane_reward + rotation_penalty

    reward = -1 if kwargs['done'] else reward
    return reward, {}

def initial_dynamics_fn(x, y, yaw, steering, speed):
    x_perturbation = 1
    yaw_perturbation = .001
    return [
        x + random.uniform(-x_perturbation,x_perturbation),
        y,
        yaw + random.uniform(-yaw_perturbation,yaw_perturbation),
        steering,
        speed,
    ]

class VistaEnv(gym.Env):
    """ This class defines a simple lane following task in Vista. It basically
    handles vehicle state update of the ego car, rendering of specified sensors,
    and determing reward and terminal condition. The default terminal condition
    is set to (1) being out of lane (2) exceed maximal roation (3) reaching the
    end of the trace.

    Args:
        trace_paths (List[str]): A list of trace paths.
        trace_config (Dict): Configuration of the trace.
        car_configs (List[Dict]): Configuration of ``every`` cars.
        sensors_configs (List[Dict]): Configuration of sensors on ``every`` cars.
        task_config (Dict): Configuration of the task, which specifies reward function
                            and terminal condition. For more details, please check the
                            source code.
        logging_level (str): Logging level (``DEBUG``, ``INFO``, ``WARNING``,
                             ``ERROR``, ``CRITICAL``); default set to ``WARNING``.

    """
    DEFAULT_CONFIG = {
        'reward_fn': lane_reward_fn,
        'terminal_condition': default_terminal_condition,
    }
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict,
                 car_config: Dict, 
                 display_config: Dict,
                 preprocess_config: Dict,
                 sensors_configs: Optional[List[Dict]] = [],
                 task_config: Optional[Dict] = dict(),
                 logging_level: Optional[str] = 'WARNING'):
        super().__init__()
        
        logging.setLevel(getattr(logging, logging_level))
        self._config = misc.merge_dict(task_config, self.DEFAULT_CONFIG)
        self._world: World = World(trace_paths, trace_config)
        self._display = Display(self._world, display_config=display_config)
        self._width, self._height = 0, 0
        
        agent = self._world.spawn_agent(car_config)
        for sensor_config in sensors_configs:
            sensor_type = sensor_config.pop('type')
            if sensor_type == 'camera':
                self._width = sensor_config['size'][0]
                self._height = sensor_config['size'][1]
                agent.spawn_camera(sensor_config)
            elif sensor_type == 'event_camera':
                agent.spawn_event_camera(sensor_config)
            elif sensor_type == 'lidar':
                agent.spawn_lidar(sensor_config)
            else:
                raise NotImplementedError(
                    f'Unrecognized sensor type {sensor_type}')

        self._distance = 0
        self._prev_xy = np.zeros((2, ))
        self._prev_yaw = 0.0
        
        self._preprocess_config = preprocess_config
        if self._preprocess_config['crop_roi']:
            i1, j1, i2, j2 = self._world.agents[0].sensors[0].camera_param.get_roi()
            self._width, self._height = i2-i1, j2-j1

        use_seqvit = True
        if use_seqvit:
            self._width = 128
            self._height = 128
            
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self._width, self._height), # change just for seq vit
                                            dtype=np.uint8)
        self.action_space = spaces.Box(low=-1/5.0, high=1/5.0, shape=(1,), dtype=np.float32)

    def _preprocess(self, image):
        # Extract ROI
        i1, j1, i2, j2 = self._world.agents[0].sensors[0].camera_param.get_roi()
        obs = image[i1:i2, j1:j2]
        obs = resize(obs, (128, 128)) # for SeqVit
        return obs

    def reset(self, seed=1, options=None):
        super().reset(seed=seed, options=options)
        
        # self._world.set_seed(seed)
        self.set_seed(seed)
        self._world.reset({self._world.agents[0].id: initial_dynamics_fn})
        # self._world.reset()
        self._display.reset()
        agent = self._world.agents[0]
        observations = self._append_agent_id(agent.observations)
        self._distance = 0
        self._prev_xy = np.zeros((2, ))
        self._prev_yaw = 0.0

        # Set info
        info = {}
        info['out_of_lane'] = False
        info['exceed_rot'] = False
        info['distance'] = self._distance
        info['agent_done'] = False

        observation = observations[agent.id]['camera_front']
        observation = self._preprocess(observation)
        observation = np.transpose(observation, (2,0,1))

        return observation, info

    def step(self, action, dt = 1/30.0):
        # Step agent and get observation
        agent = self._world.agents[0]

        action = np.array([action[0], agent.human_speed])
        agent.step_dynamics(action, dt=dt)
        agent.step_sensors()
        observations = agent.observations
        observation = observations['camera_front']
        observation = self._preprocess(observation)
        observation = np.transpose(observation, (2,0,1))

        # Define terminal condition
        done, info_from_terminal_condition = self.config['terminal_condition'](
            self, agent.id)

        # Get info
        info = misc.fetch_agent_info(agent)
        info['out_of_lane'] = info_from_terminal_condition['out_of_lane']
        info['exceed_max_rot'] = info_from_terminal_condition['exceed_max_rot']
        info['agent_done'] = info_from_terminal_condition['agent_done']

        # Define reward
        reward, _ = self.config['reward_fn'](self, agent.id, self._prev_yaw,
                                             **info_from_terminal_condition)

        current_xy = agent.ego_dynamics.numpy()[:2]
        # print(f"curr xy: {current_xy}")
        # print(f"prev xy: {self._prev_xy}")
        self._distance += np.linalg.norm(current_xy - self._prev_xy)
        self._prev_xy = current_xy

        # print(f"prev yaw: {self._prev_yaw}")
        # print(f"current yaw: {agent.ego_dynamics.numpy()[2]}\n")
        self._prev_yaw = agent.ego_dynamics.numpy()[2]
    
        info['distance'] = self._distance

        truncated = False
        # Pack output
        # observation, reward, done, info = map(
        #     self._append_agent_id, [observation, reward, done, info])
        # print(observation)

        return observation, reward, done, truncated, info

    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.
        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.world.set_seed(seed)

    def _append_agent_id(self, data):
        agent = self._world.agents[0]
        return {agent.id: data}

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "rgb_array":
            agent = self._world.agents[0]
            observations = self._append_agent_id(agent.observations)
            observation = observations['camera_front']
            observation = self._preprocess(observation)
            observation = np.transpose(observation, (2,0,1))
            return observation

    def close(self):
        pass

    @property
    def config(self) -> Dict:
        """ Configuration of this task. """
        return self._config

    @property
    def world(self) -> World:
        """ :class:`World` of this task. """
        return self._world

    @property
    def seed(self) -> int:
        """ Random seed for the task and the associated :class:`World`. """
        return self._seed