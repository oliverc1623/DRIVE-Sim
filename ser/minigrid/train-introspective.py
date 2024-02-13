import os
import glob
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import numpy as np

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from IntrospectiveEnv import IntrospectiveEnv, IntrospectiveEnvLocked, SmallUnlockedDoorEnv

from PPO_Introspective import PPOIntrospective
from introspective import introspect, correct

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "Empty8x8EnvStudent"

    size=8 # gridworld env size

    has_continuous_action_space = False  # continuous action space; else discrete
    save_frames = False

    max_ep_len = 4 * size**2                   # max timesteps in one episode
    max_training_timesteps = int(5e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 5        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 4               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0005       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 46         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = SmallUnlockedDoorEnv(size=size, render_mode="rgb_array", locked=True)
    print(f"Gridworld size: {env.max_steps}")

    # state space dimension
    state_dim = env.observation_space['image'].shape[2]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 3      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    student_ppo_agent = PPOIntrospective(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, teacher=False)
    teacher_ppo_agent = PPOIntrospective(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, teacher=True)
    # TODO: we need to fine-tune a copy 
    teacher_ppo_agent.policy.load_state_dict(torch.load("PPO_preTrained/Empty5x5Env/PPO_Empty5x5Env_46_2.pth"))
    teacher_ppo_agent.policy_old.load_state_dict(torch.load("PPO_preTrained/Empty5x5Env/PPO_Empty5x5Env_46_2.pth"))


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_running_advice = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    advice_given = 0

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()
        state = state["image"]
        
        current_ep_reward = 0
        current_advice_given = 0

        for t in range(1, max_ep_len+1):

            if time_step % 500000 == 0:
                random_seed += 1
                print("--------------------------------------------------------------------------------------------")
                print("setting random seed to ", random_seed)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)

            # select action with policy
            h = introspect(teacher_ppo_agent.preprocess(state), teacher_ppo_agent.policy_old, teacher_ppo_agent.policy, time_step, inspection_threshold=0.9)
            if h:
                action, teacher_state, teacher_action_logprob, teacher_state_val = teacher_ppo_agent.select_action(state)
                student_ppo_agent.buffer.actions.append(action)
                student_ppo_agent.buffer.states.append(teacher_state)
                student_ppo_agent.buffer.logprobs.append(teacher_action_logprob)
                student_ppo_agent.buffer.state_values.append(teacher_state_val)
                current_advice_given += 1
            else:
                action, state, action_logprob, state_val = student_ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action.item())
            state = state["image"]

            # saving reward and is_terminals
            student_ppo_agent.buffer.rewards.append(reward)
            student_ppo_agent.buffer.is_terminals.append(done)
            student_ppo_agent.buffer.indicators.append(h)

            if save_frames:
                img = env.render()
                plt.imsave(f"frames/frame_{time_step:06}.png", img)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                teacher_correction, student_correction = correct(student_ppo_agent.buffer, student_ppo_agent.policy_old, teacher_ppo_agent.policy_old)
                teacher_ppo_agent.update_critic(teacher_correction, student_ppo_agent.buffer)
                student_ppo_agent.update(student_correction)
                # print(f"teacher policy old critic: {teacher_ppo_agent.policy_old.critic[10].weight[0].sum()}")
                # print(f"teacher policy critic: {teacher_ppo_agent.policy.critic[10].weight[0].sum()}")

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                student_ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print_avg_advice = print_running_advice / print_running_episodes
                print_avg_advice = round(print_avg_advice, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Average Advice given: {}".format(i_episode, 
                                                                                                                      time_step, 
                                                                                                                      print_avg_reward, 
                                                                                                                      print_avg_advice))

                print_running_reward = 0
                print_running_episodes = 0
                print_running_advice = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                student_ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        print_running_advice += current_advice_given

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()