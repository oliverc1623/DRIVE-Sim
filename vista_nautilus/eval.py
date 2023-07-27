import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../vista_nautilus/models/')
import convlstm2
import mycnn
import a2c_cnn
import vista
from helper import *
import datetime
import argparse


device = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(f"Using {device} device")

def test(name, model, world, car, display, camera, device, f):
    print(f"Testing {name}...")
    world.set_seed(47)
    vista_reset(world, display)
    score = 0.0
    num_test = 5
    trace_index = car.trace_index
    save_flag = False
    for _ in range(num_test):
        step = 0
        world.set_seed(47)
        vista_reset(world, display)
        observation = grab_and_preprocess_obs(car, camera).to(device)
        done = False
        prev_curvature = 0.0
        trace_index = car.trace_index
        initial_frame = car.frame_index
        total_frames = len(car.trace.good_frames['camera_front'][0])
        track_left = total_frames - initial_frame
        episode_reward = 0.0
        while not done:
            if name == "A2C":
                mu, sigma = model.pi(observation.permute(2,0,1))
            elif name == "REINFORCE":
                observation = observation.unsqueeze(0).permute(0,3,1,2)
                mu, logsigma = model(observation)
                mu = 1/8.0 * torch.tanh(mu)  # conversion
                sigma = 0.1 * torch.sigmoid(logsigma) + 0.005  # conversion
            dist = Normal(mu, sigma)
            action = dist.sample().item()
            vista_step(car, action)
            prev_curvature = action
            observation_prime = grab_and_preprocess_obs(car, camera).to(device)
            reward = calculate_reward(car, action, prev_curvature)
            episode_reward += reward
            done = int(check_crash(car))

            observation = observation_prime
            score += reward
            step += 1

        print(f"steps: {step}")
        progress = car.frame_index - initial_frame
        progress_percentage = np.round(progress/track_left, 4)
        # Write reward and loss to results txt file
        f.write(f"{name}\t{episode_reward}\t{step}\t{trace_index}\t{car.done}\t{progress_percentage}\n")
        f.flush()
        done = False
    print(f"Model: {name}, avg score : {score/num_test:.1f}\n")
    return score/num_test # return avg score

def main(args):
    # REINFORCE
    reinforce_model = mycnn.CNN().to(device)
    reinforce_cnn_dict = torch.load('saved_models/REINFORCE_CNN_model.pth')
    reinforce_model.load_state_dict(reinforce_cnn_dict['model_state_dict'])
    reinforce_model.eval()

    # A2C
    a2c_model = a2c_cnn.ActorCritic().to(device)
    a2c_cnn_dict = torch.load('saved_models/best_a2c_model_checkpoint.pth')
    a2c_model.load_state_dict(a2c_cnn_dict['model_state_dict'])
    a2c_model.eval()

    # TODO: PPO

    # Start VISTA Environment
    trace_root = "vista_traces"
    trace_path = [
        "20210726-154641_lexus_devens_center",
        "20210726-155941_lexus_devens_center_reverse",
        "20210726-184624_lexus_devens_center",
        "20210726-184956_lexus_devens_center_reverse",
    ]
    trace_path = [os.path.join(trace_root, p) for p in trace_path]
    world = vista.World(trace_path, trace_config={"road_width": 4})
    car = world.spawn_agent(config={"length": 5.0,
                                    "width": 2.0,
                                    "wheel_base": 2.78,
                                    "steering_ratio": 14.7,
                                    "lookahead_road": True,})
    camera = car.spawn_camera(config={"size": (355, 413)})
    display = vista.Display(
        world, display_config={"gui_scale": 2, "vis_full_frame": False}
    )
    world.set_seed(47)

    # Evaluation Hyperparameters
    models = [("A2C", a2c_model), ("REINFORCE", reinforce_model)]

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = "eval_results"
    model_results_dir = results_dir

    # if self.animate and not os.path.exists(self.model_frame_dir):
    #     os.makedirs(self.model_frame_dir)

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)

    # Define the file path
    filename = args.filename
    print("Writing log to: " + filename + ".txt")
    file_path = os.path.join(model_results_dir, filename + ".txt")
    f = open(file_path, "w")
    f.write("model\treward\tsteps\ttrace\tdone\tcompleted\n")

    for name, model in models:
        test(name, model, world, car, display, camera, device, f)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="VISTA DRL Evaluator")
    parser.add_argument("-f", "--filename", required=True)
    args = parser.parse_args()
    main(args)