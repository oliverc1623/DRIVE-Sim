import vista
import torch
from torch import nn
import os
from data_preprocessing import *
from terminal import * 
from NeuralNetwork import run_driving_model, compute_driving_loss
import matplotlib.pyplot as plt


trace_root = "../trace"
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

driving_model = nn.Sequential(
    nn.Conv2d(3, 32, 5),
    nn.SiLU(),
    nn.Conv2d(32, 48, 5),
    nn.SiLU(),
    nn.Conv2d(48, 64, 3),
    nn.SiLU(),
    nn.Conv2d(64, 64, 3),
    nn.SiLU(),
    nn.Flatten(),
    nn.Linear(302016, 128),
    nn.SiLU(),
    nn.Linear(128, 2),
)  # NeuralNetwork()
driving_model.load_state_dict(torch.load("model.pth"))
driving_model.eval()

## Evaluation block!##

i_step = 0
num_episodes = 5
num_reset = 5
for i_episode in range(num_episodes):
    # Restart the environment
    vista_reset()
    observation = grab_and_preprocess_obs(car, camera)

    print("rolling out in env")
    episode_step = 0
    while not check_crash(car) and episode_step < 100:
        # using our observation, choose an action and take it in the environment
        curvature_dist = run_driving_model(observation, driving_model)
        print(f"curvature distribution: {curvature_dist}")
        # curvature = curvature_dist.loc[0][0].detach().numpy()
        curvature = curvature_dist.sample()[0, 0]
        print(f"curvature: {curvature}\n")

        # Step the simulated car with the same action
        vista_step(curvature)
        observation = grab_and_preprocess_obs(car, camera)

        vis_img = display.render()
        plt.pause(.05)
        i_step += 1
        episode_step += 1

    for _ in range(num_reset):
        i_step += 1

print(f"Average reward: {(i_step - (num_reset*num_episodes)) / num_episodes}")

print("Saving trajectory with trained policy...")
# stream.save("trained_policy.mp4")
# mdl.lab3.play_video("trained_policy.mp4")
