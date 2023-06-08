import argparse
import gym
import os
import sys
import pickle
import time
import vista
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.vista_ac import VistaAC
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.vista_helper import * 


parser = argparse.ArgumentParser(description='PyTorch A2C for VISTA')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float32
torch.set_default_dtype(dtype)
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device)

"""environment"""
trace_root = "../../../trace"
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
display = vista.Display(
    world, display_config={"gui_scale": 2, "vis_full_frame": False}
)

"""seeding"""
world.set_seed(47)

"""define actor and critic"""
policy_net = VistaAC()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=0.01)

"""create agent"""
agent = Agent(num_threads=args.num_threads)


def update_params(batch):
    print("in update params")
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = policy_net.v(states)

    """get advantage estimation from the trajectories"""
    advantages, td_target = calculate_reward(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    a2c_step(policy_net, optimizer_policy, states, actions, td_target, advantages, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        print(f"Episode: {i_iter}")
        """generate multiple trajectories that reach the minimum batch_size"""
        print("before collecting training samples")
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        print("done collecting training samples")
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        """evaluate with determinstic action (remove noise for exploration)"""
        print("going in eval")
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
