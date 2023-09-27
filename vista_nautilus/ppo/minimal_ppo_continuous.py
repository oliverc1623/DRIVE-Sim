# credit to: https://github.com/seolhokim
# import gymnasium as gym
import os
os.environ["DISPLAY"] = ":1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging, misc
from vista.tasks import MultiAgentBase
from vista.utils import transform
import sys
sys.path.insert(1, '../vista_nautilus/')
sys.path.insert(1, '../vista_nautilus/models/')
from helper import * 

#Hyperparameters
entropy_coef = 1e-2
critic_coef = 1
learning_rate = 0.000005
gamma         = 0.9
lmbda         = 0.9
eps_clip      = 0.2
K_epoch       = 10
T_horizon     = 20

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, device, channels = 3, dim_head = 64):
        super().__init__()
        
        self.data = []
        self.device = device
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)
        self.critic_head = nn.Linear(dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        x = self.linear_head(x)

        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        mu = 1/8.0 * torch.tanh(mu)  # conversion
        sigma = 0.1 * torch.sigmoid(log_sigma) + 0.005  # conversion
        return mu, sigma
    
    def v(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.critic_head(x)


    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(torch.tensor(r,dtype=torch.float32))
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s_lst = torch.stack(s_lst).to(self.device)
        r_lst = torch.stack(r_lst).to(self.device)
        s_prime_lst = torch.stack(s_prime_lst).to(dtype=torch.float32)
        
        s = s_lst 
        a = torch.tensor(a_lst).to(self.device)
        r = r_lst 
        s_prime = s_prime_lst.to(self.device)
        done_mask = torch.tensor(done_lst).to(self.device)
        prob_a = torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
        # print(f"r shape: {r.shape}")
        for i in range(K_epoch):
            # print(f"s_prime shape: {s_prime.shape}")
            td_target = r.unsqueeze(1) + gamma * self.v(s_prime.permute(0,3,1,2)) * done_mask
            delta = td_target - self.v(s.permute(0,3,1,2))
            delta = torch.flip(delta, (0,)) #delta

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32).to(self.device)
            
            # print(f"s shape: {s.shape}")
            curr_mu,curr_sigma = self.forward(s.permute(0,3,1,2))
            
            curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
            curr_log_prob = curr_dist.log_prob(a)
            entropy = curr_dist.entropy() * entropy_coef
            
            ratio = torch.exp(curr_log_prob - old_log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
            v = self.v(s.permute(0,3,1,2)).float()
            td = td_target.detach().float()
            critic_loss = critic_coef * F.smooth_l1_loss(v, td)
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            


def main(render = False):
    if torch.cuda.is_available():
        device= 'cuda:0'
    else:
        device = 'cpu'
    print('device:{}'.format(device))
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
            size=(400, 640), # (200, 320) for lighter cnn, (355, 413) for 80x200
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
    ]
    task_config = dict(n_agents=2,
                        mesh_dir="carpack01",
                        init_dist_range=[50., 60.],
                        init_lat_noise_range=[-3., 3.],
                        reward_fn=my_reward_fn)
    display_config = dict(road_buffer_size=1000, )

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
    env = MultiAgentBase(trace_paths=trace_path,
                            trace_config=trace_config,
                            car_configs=[car_config] * task_config['n_agents'],
                            sensors_configs=[sensors_config] + [[]] *
                            (task_config['n_agents'] - 1),
                            task_config=task_config)
    display = vista.Display(env.world, display_config=display_config)

    # start env
    env.world.set_seed(47)
    env.reset();
    display.reset()
    model =  SimpleViT(
        image_size = 144,
        patch_size = 16,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        device = device,
        heads = 16,
        mlp_dim = 2048
    ).to(device)

    import cv2
    # open and write to file to track progress
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = "results"
    model_results_dir = results_dir + "/CNN/" # TODO: make this into variable

    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    # Define the file path
    filename = "ppo_vit_trial4"
    print("Writing log to: " + filename + ".txt")
    file_path = os.path.join(model_results_dir, filename + ".txt")
    f = open(file_path, "w")
    f.write("reward\tsteps\tprogress\ttrace\n")

    print_interval = 1
    score = 0.0
    global_step = 0
    best_score = float("-inf")
 
    for n_epi in range(250):
        print(f"n_episode: {n_epi}")
        env.world.set_seed(47)
        observation = env.reset();
        observation = grab_and_preprocess_obs(observation, env, device)
        done = False
        total_progress = 0.0
        steps = 0
        score = 0.0
        trace_index = env.ego_agent.trace_index
        initial_frame = env.ego_agent.frame_index
        while not done: 
            for t in range(T_horizon):
                global_step += 1 
                mu,sigma = model(observation.permute(2,0,1).unsqueeze(0).to(device))
                dist = torch.distributions.Normal(mu,sigma)
                actions = sample_actions(dist, env.world, env.ego_agent.id)
                a = torch.tensor([[actions[env.ego_agent.id][0]]], dtype=torch.float32).to(device)
                log_prob = dist.log_prob(a)
                observation_prime, rewards, dones, infos = env.step(actions)
                observation_prime = grab_and_preprocess_obs(observation_prime, env, device)
                done = rewards[env.ego_agent.id][1]['done']
                reward = 0.0 if done else rewards[env.ego_agent.id][0]
                model.put_data((observation, a, reward, observation_prime, \
                                log_prob, done))
                observation = observation_prime
                score += reward
                steps += 1
                if done:
                    break
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            progress = calculate_progress(env, initial_frame)
            print(f"# of episode :{n_epi}, score : {score/print_interval:.1f}, steps : {steps}, progress : {progress*100}%")
            f.write(f"{score}\t{steps}\t{progress}\t{trace_index}\n")
            f.flush()

            if score > best_score:
                best_score = score
                print("Saving and exporting model...")
                checkpoint = {
                    'epoch': n_epi,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'best_accuracy': score,
                }
                torch.save(checkpoint, f"saved_models/ppo_cnn_model_{timestamp}.pth")

main()