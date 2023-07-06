import vista
import torch
import os
import matplotlib.pyplot as plt
from REINFORCE import mycnn
from my_ppo.vista_helper import *
import torch.distributions as dist


trace_root = "trace"
trace_path = [
    "20210726-154641_lexus_devens_center",
    "20210726-155941_lexus_devens_center_reverse",
    "20210726-184624_lexus_devens_center",
    "20210726-184956_lexus_devens_center_reverse",
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]
world = vista.World(trace_path, trace_config={"road_width": 4})
car = world.spawn_agent(
    config={
        "length": 5.0,
        "width": 2.0,
        "wheel_base": 2.78,
        "steering_ratio": 14.7,
        "lookahead_road": True,
    }
)
camera = car.spawn_camera(config={"size": (200, 320)})
display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": False})


def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None:
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None:
        speed = car.trace.f_speed(car.timestamp)
    print(f"speed: {speed}")
    car.step_dynamics(action=np.array([curvature, speed]), dt=1 / 15.0)
    car.step_sensors()

def vista_reset():
    world.reset()
    display.reset()

vista_reset()

def run_driving_model(image, model):
    max_curvature=1/8.0
    max_std=0.1
    single_image_input = len(image.shape) == 3  # missing 4th batch dimension
    if single_image_input:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    mu, logsigma = model(image)
    mu = max_curvature * torch.tanh(mu)  # conversion
    sigma = max_std * torch.sigmoid(logsigma) + 0.005  # conversion

    pred_dist = dist.Normal(mu, sigma)
    return pred_dist


driving_model = mycnn.CNN()
driving_model.load_state_dict(torch.load("models/reinforce1_2023-07-05_17-18-02_.pth"))

print(driving_model)

## Evaluation block!##

i_step = 0
num_episodes = 5
num_reset = 5
for i_episode in range(num_episodes):
    # Restart the environment
    # world.set_seed(47)
    vista_reset()
    observation = grab_and_preprocess_obs(car, camera)

    print("rolling out in env")
    episode_step = 0
    while not check_crash(car):
        curvature_dist = run_driving_model(observation, driving_model)
        curvature_action = curvature_dist.sample()[0, 0]
        curvature_action = curvature_action.cpu().detach()
        vista_step(curvature_action)
        observation = grab_and_preprocess_obs(car, camera)

        vis_img = display.render()
        plt.pause(.05)
        i_step += 1
        episode_step += 1

    for _ in range(num_reset):
        i_step += 1

print(f"Average reward: {(i_step - (num_reset*num_episodes)) / num_episodes}")