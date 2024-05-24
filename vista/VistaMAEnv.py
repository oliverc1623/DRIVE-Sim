import gymnasium as gym
from gymnasium import spaces

from typing import Optional, List, Dict, Any
import numpy as np
from shapely.geometry import box as Box
from shapely import affinity
import random 
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from vista import World
from vista.entities.agents.Car import Car
from vista.entities.agents.Dynamics import StateDynamics
from vista.entities.sensors.MeshLib import MeshLib
from vista.utils import logging, misc, transform


def default_terminal_condition(task, agent_id, **kwargs):
    """ An example definition of terminal condition. """

    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]

    def _check_out_of_lane():
        road_half_width = agent.trace.road_width / 2.
        return np.abs(agent.relative_state.x) > road_half_width

    def _check_exceed_max_rot():
        maximal_rotation = np.pi / 10.
        return np.abs(agent.relative_state.yaw) > maximal_rotation

    def _check_crash():
        other_agents = [_a for _a in task.world.agents if _a.id != agent_id]
        agent2poly = lambda _x: misc.agent2poly(
            _x, ref_dynamics=agent.human_dynamics)
        poly = agent2poly(agent)
        other_polys = list(map(agent2poly, other_agents))
        overlap = compute_overlap(poly, other_polys) / poly.area
        crashed = np.any(overlap > task.config['overlap_threshold'])
        return crashed

    out_of_lane = _check_out_of_lane()
    exceed_max_rot = _check_exceed_max_rot()
    crashed = _check_crash()
    done = out_of_lane or exceed_max_rot or crashed or agent.done
    other_info = {
        'done': done,
        'out_of_lane': out_of_lane,
        'exceed_max_rot': exceed_max_rot,
        'crashed': crashed,
        'agent_done': agent.done
    }

    return done, other_info

def get_rotation_penalty(agent_orientation, target_orientation, max_rotation_penalty):
    orientation_difference = abs(agent_orientation - target_orientation)
    rotation_threshold = 0.01  # Adjust this threshold as needed
    rotation_penalty = max(0, orientation_difference - rotation_threshold)
    penalty = -rotation_penalty * max_rotation_penalty
    return penalty

def default_reward_fn(task, agent_id, prev_yaw, **kwargs):
    agent = [_a for _a in task.world.agents if _a.id == agent_id][0]
    other_agents = [_a for _a in task.world.agents if _a.id != agent_id]

    # compute lane reward
    road_width = agent.trace.road_width
    z_lat = road_width / 2
    q_lat = np.abs(agent.relative_state.x)
    lane_reward = 1 - (q_lat/z_lat)**2

    # compute rotation penalty
    rotation_penalty = get_rotation_penalty(prev_yaw, agent.ego_dynamics.numpy()[2], 1)

    # compute speed reward
    speed_reward = 1 - abs(agent.speed - agent.human_speed)/15

    # collision avoidance reward
    agent2poly = lambda _x: misc.agent2poly(
        _x, ref_dynamics=agent.human_dynamics)
    poly = agent2poly(agent).buffer(5)
    other_polys = list(map(agent2poly, other_agents))
    overlap = (compute_overlap(poly, other_polys) / poly.area) * 1    

    reward = lane_reward + rotation_penalty + speed_reward - overlap[0]
    reward = -1 if kwargs['done'] else reward
    return reward, {}

def initial_dynamics_fn(x, y, yaw, steering, speed):
    x_perturbation = 1.5
    yaw_perturbation = .1
    return [
        x + random.uniform(-x_perturbation,x_perturbation),
        y,
        yaw + random.uniform(-yaw_perturbation,yaw_perturbation),
        steering,
        speed,
    ]

class VistaMAEnv(gym.Env):
    """ This class builds a simple environment with multiple cars in the scene, which
    involves randomly initializing ado cars in the front of the ego car, checking collision
    between cars, handling meshes for all virtual agents, and determining terminal condition.

    Args:
        trace_paths (List[str]): A list of trace paths.
        trace_config (Dict): Configuration of the trace.
        car_configs (List[Dict]): Configuration of ``every`` cars.
        sensors_configs (List[Dict]): Configuration of sensors on ``every`` cars.
        task_config (Dict): Configuration of the task. An example (default) is,

            >>> DEFAULT_CONFIG = {
                    'n_agents': 1,
                    'mesh_dir': None,
                    'overlap_threshold': 0.05,
                    'max_resample_tries': 10,
                    'init_dist_range': [5., 10.],
                    'init_lat_noise_range': [-1., 1.],
                    'init_yaw_noise_range': [-0.0, 0.0],
                    'reward_fn': default_reward_fn,
                    'terminal_condition': default_terminal_condition
                }

            Note that both ``reward_fn`` and ``terminal_condition`` have function signature
            as ``f(task, agent_id, **kwargs) -> (value, dict)``. For more details, please check
            the source code.
        logging_level (str): Logging level (``DEBUG``, ``INFO``, ``WARNING``,
                             ``ERROR``, ``CRITICAL``); default set to ``WARNING``.

    """
    DEFAULT_CONFIG = {
        'n_agents': 1,
        'mesh_dir': None,
        'overlap_threshold': 0.05,
        'max_resample_tries': 10,
        'init_dist_range': [5., 10.],
        'init_lat_noise_range': [-1., 1.],
        'init_yaw_noise_range': [-0.0, 0.0],
        'reward_fn': default_reward_fn,
        'terminal_condition': default_terminal_condition
    }

    def __init__(self,
                 trace_paths: List[str],
                 trace_config: Dict,
                 car_configs: List[Dict],
                 sensors_configs: List[List[Dict]],
                 preprocess_config: Dict,
                 task_config: Optional[Dict] = dict(),
                 logging_level: Optional[str] = 'WARNING'):
        logging.setLevel(getattr(logging, logging_level))
        self._config = misc.merge_dict(task_config, self.DEFAULT_CONFIG)
        n_agents = self.config['n_agents']
        assert len(
            car_configs
        ) == n_agents, 'Number of car config is not consistent with number of agents'
        assert len(
            sensors_configs
        ) == n_agents, 'Number of sensors config is not consistent with number of agents'
        assert car_configs[0][
            'lookahead_road'], '\'lookahead_road\' in the first car config should be set to True'

        self._world: World = World(trace_paths, trace_config)
        # self._display = Display(self._world)
        self._width, self._height = 0, 0
        self.render_mode = "rgb_array"
        for i in range(n_agents):
            agent = self._world.spawn_agent(car_configs[i])
            for sensor_config in sensors_configs[i]:
                sensor_type = sensor_config.pop('type')
                if sensor_type == 'camera':
                    self._width = sensor_config['size'][0]
                    self._height = sensor_config['size'][1]
                    agent.spawn_camera(sensor_config)
                else:
                    raise NotImplementedError(
                        f'Unrecognized sensor type {sensor_type}')
        if n_agents > 1:
            assert self.config[
                'mesh_dir'] is not None, 'Specify mesh_dir if n_agents > 1'
            self._meshlib = MeshLib(self.config['mesh_dir'])
        self.set_seed(0)
        self._distance = 0
        self._prev_xy = np.zeros((2, ))
        self._prev_yaw = 0.0
        self._preprocess_config = preprocess_config
        self._is_seq = preprocess_config['seq']
        if self._preprocess_config['crop_roi']:
            i1, j1, i2, j2 = self._world.agents[0].sensors[0].camera_param.get_roi()
            self._width, self._height = i2-i1, j2-j1
        if self._preprocess_config['resize']:
            self._width = 84
            self._height = 84
        obs_shape = (3, self._width, self._height)
        if self._preprocess_config['grayscale']:
            obs_shape = (1, self._width, self._height)
        else:
            obs_shape = (3, self._width, self._height)
        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=np.array([-0.75, 1.0]), 
            high=np.array([0.75, 15.0]), 
            shape=(2,), 
            dtype=np.float32
        )

    def _get_course_completion_rate(self):
        cur_frame = self._world.agents[0].frame_index
        trace_index = self._world.agents[0].trace_index
        num_frames = self._world.traces[trace_index].num_of_frames
        frames_left = num_frames - cur_frame
        course_completion_rate = (self._distance/frames_left)*100
        return round(course_completion_rate, 4)

    def _preprocess(self, image, seq=False):
        # grayscale
        if self._preprocess_config['grayscale']:
            image = rgb2gray(image)
            image = (image*255).astype('uint8')
        # cropped ROI
        i1, j1, i2, j2 = self._world.agents[0].sensors[0].camera_param.get_roi()
        image = image[i1:i2, j1:j2]
        # resize
        if self._preprocess_config['resize']:
            image = resize(image, (self._width, self._height), anti_aliasing=True)
            image = (image*255).astype('uint8')
        # binarize
        if self._preprocess_config['binary']:
            thresh = threshold_otsu(image)
            image = image > thresh
        image = np.expand_dims(image, axis=0)
        return image


    def reset(self, seed=1, options=None) -> Dict:
        super().reset(seed=seed, options=options)
        # Reset world; all agents are initialized at the same pointer to the trace
        new_trace_index, new_segment_index, new_frame_index = \
            self.world.sample_new_location()
        for agent in self.world.agents:
            if agent == self.ego_agent:
                agent.reset(
                    new_trace_index,
                    new_segment_index,
                    new_frame_index,
                    initial_dynamics_fn = initial_dynamics_fn,
                    step_sensors=False
                )
            else:
                agent.reset(
                    new_trace_index,
                    new_segment_index,
                    new_frame_index,
                    step_sensors=False
                )
        # Randomly initialize ado agents in the front
        ref_dynamics = self.ego_agent.human_dynamics
        polys = [misc.agent2poly(self.ego_agent, ref_dynamics)]
        for agent in self.world.agents:
            if agent == self.ego_agent:
                continue
            collision_free = False
            resample_tries = 0
            while not collision_free and resample_tries < self.config[
                    'max_resample_tries']:
                self._randomly_place_agent(agent)
                poly = misc.agent2poly(agent, ref_dynamics)
                overlap = compute_overlap(poly, polys) / poly.area
                collision_free = np.all(
                    overlap <= self.config['overlap_threshold'])

                resample_tries += 1
            polys.append(poly)

        # Reset mesh library
        if len(self.world.agents) > 1:
            self._reset_meshlib()

        # Set info
        info = {}
        info['out_of_lane'] = False
        info['exceed_rot'] = False
        info['distance'] = self._distance
        info['agent_done'] = False
        info['crashed'] = False
        info['course_completion_rate'] = 0.0

        # Get observation
        self._sensor_capture()
        # observations = {_a.id: _a.observations for _a in self.world.agents}
        observations = self.ego_agent.observations
        observation = observations['camera_front']
        observation = self._preprocess(observation)
        self._distance = 0
        self._prev_xy = np.zeros((2, ))
        self._prev_yaw = 0.0
        return observation, info


    def step(self, action, dt=1 / 30.):
        """ Step the environment. This includes updating agents' states, synthesizing
        agents' observations, checking terminal conditions, and computing rewards.

        Args:
            actions (Dict[str, np.ndarray]):
                A dictionary with keys as agent IDs and values as actions
                to be executed to interact with the environment and other
                agents.
            dt (float): Elapsed time in second; default set to 1/30.

        Returns:
            Return a tuple (``dict_a``, ``dict_b``, ``dict_c``, ``dict_d``),
            where ``dict_a`` is the observation, ``dict_b`` is the reward,
            ``dict_c`` is whether the episode terminates, ``dict_d`` is additional
            informations for every agents; keys of every dictionary are agent IDs.

        """
        # Update agents' dynamics (state)
        for agent in self.world.agents:
            if agent != self.ego_agent:
                botaction = np.array([0.0, 0.0])
                agent.step_dynamics(botaction, dt=dt)

        action = np.array([action[0], action[1]])
        self.ego_agent.step_dynamics(action, dt=dt)
        self.ego_agent.step_sensors()

        # Get agents' sensory measurement
        self._sensor_capture()
        observations = self.ego_agent.observations
        observation = observations['camera_front']
        observation = self._preprocess(observation)

        # Check terminal conditions
        dones = dict()
        infos_from_terminal_condition = dict()
        terminal_condition = self.config['terminal_condition']
        for agent in self.world.agents:
            dones[agent.id], infos_from_terminal_condition[
                agent.id] = terminal_condition(self, agent.id)

        # Compute reward
        rewards = dict()
        reward_fn = self.config['reward_fn']
        for agent in self.world.agents:
            rewards[agent.id], _ = reward_fn(
                self, agent.id, self._prev_yaw, **infos_from_terminal_condition[agent.id])

        # metrics
        current_xy = self.ego_agent.ego_dynamics.numpy()[:2]
        self._distance += 1 # np.linalg.norm(current_xy - self._prev_xy)
        self._prev_xy = current_xy
        self._prev_yaw = self.ego_agent.ego_dynamics.numpy()[2]

        # Get info
        # infos = dict()
        info = misc.fetch_agent_info(self.ego_agent)
        ego_id = self.ego_agent.id
        info['out_of_lane'] = infos_from_terminal_condition[ego_id]['out_of_lane']
        info['exceed_max_rot'] = infos_from_terminal_condition[ego_id]['exceed_max_rot']
        info['agent_done'] = infos_from_terminal_condition[ego_id]['agent_done']
        info['crashed'] = infos_from_terminal_condition[ego_id]['crashed']
        info['distance'] = self._distance
        info['course_completion_rate'] = self._get_course_completion_rate()
        truncated=False
        if np.floor(info['course_completion_rate']) == 100.0:
            truncated = True
        return observation, rewards[self.ego_agent.id], dones[self.ego_agent.id], truncated, info

    
    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.

        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.world.set_seed(seed)


    def _randomly_place_agent(self, agent: Car):
        # Randomly sampled a pose in the front of ego agent that is still on
        # the road. This can be achieved by,
        # (1) randomly sampling a distance from the ego agent
        # (2) fetch the closest pointer from the road kept by the ego agent
        # (3) slightly perturb the associated pose.
        tgt_dist = self._rng.uniform(*self.config['init_dist_range'])

        road = np.array(self.ego_agent.road)
        dist_from_ego = np.linalg.norm(road[:, :2], axis=1)
        tgt_idx = np.argmin(np.abs(tgt_dist - dist_from_ego))
        tgt_pose = road[tgt_idx].copy()

        lat_noise = self._rng.uniform(*self.config['init_lat_noise_range'])
        tgt_pose[0] += lat_noise * np.cos(tgt_pose[2])
        tgt_pose[1] += lat_noise * np.sin(tgt_pose[2])
        yaw_noise = self._rng.uniform(*self.config['init_yaw_noise_range'])
        tgt_pose[2] += yaw_noise

        # Place agent given the randomly sampled pose
        agent.ego_dynamics.update(*tgt_pose)
        agent.step_dynamics(tgt_pose[-2:], dt=1e-8)

    def _reset_meshlib(self):
        self._meshlib.reset(self.config['n_agents'])

        # Assign car width and length based on mesh size
        for i, agent in enumerate(self.world.agents):
            agent._width = self._meshlib.agents_meshes_dim[i][0]
            agent._length = self._meshlib.agents_meshes_dim[i][1]

    def _sensor_capture(self) -> None:
        # Update mesh of virtual agents
        if self.config['n_agents'] > 1:
            for agent in self.world.agents:
                if len(agent.sensors) == 0:
                    continue

                for i, other_agent in enumerate(self.world.agents):
                    if other_agent.id == agent.id:
                        continue
                    # compute relative pose to the ego agent
                    latlongyaw = transform.compute_relative_latlongyaw(
                        other_agent.ego_dynamics.numpy()[:3],
                        agent.human_dynamics.numpy()[:3])
                    # add to sensor
                    scene_object = self._meshlib.agents_meshes[i]
                    name = f'agent_{i}'
                    for sensor in agent.sensors:
                        sensor.update_scene_object(name, scene_object,
                                                   latlongyaw)

        # Step sensors
        for agent in self.world.agents:
            agent.step_sensors()

    @property
    def config(self) -> Dict:
        """ Configuration of this task. """
        return self._config

    @property
    def ego_agent(self) -> Car:
        """ Ego agent. """
        return self.world.agents[0]

    @property
    def world(self) -> World:
        """ :class:`World` of this task. """
        return self._world

    @property
    def seed(self) -> int:
        """ Random seed for the task and the associated :class:`World`. """
        return self._seed

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "rgb_array":
            observations = self.ego_agent.observations
            observation = observations['camera_front']
            observation = self._preprocess(observation, self._is_seq)
            observation = np.transpose(observation, (2,0,1))
            return observation

    def close(self):
        pass


def compute_overlap(poly: Box, polys: List[Box]) -> List[float]:
    """ Compute overlapping area between 1 polygons and N polygons.

    Args:
        poly (shapely.geometry.Box): A polygon.
        poly (List): A list of polygon.

    Returns:
        List[float]: Intersecting area between polygons.

    """
    n_polys = len(polys)
    overlap = np.zeros((n_polys))
    for i in range(n_polys):
        intersection = polys[i].intersection(poly)
        overlap[i] = intersection.area
    return overlap
