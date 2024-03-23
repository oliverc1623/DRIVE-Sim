#!/usr/bin/env python
# coding: utf-8

# In[149]:


from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from typing import Optional
from typing import Tuple
import seaborn as sns
import numpy as np
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})


# In[154]:


log_dir = "ddpg/ddpg-tmp-trial4-lane-follow/"
timesteps = 100_000
plot_results([log_dir], timesteps, results_plotter.X_EPISODES, "PPO VISTA")
plt.show()


# In[113]:


def my_plot_results(
    dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    return xy_list

def xyListToDataFrame(xylist):
    names = ['Episode', 'Reward']
    df = pd.DataFrame(xylist, columns=names)
    df = df.explode(['Episode','Reward'])
    return df

def getDataFrame(log_dir):
    timesteps = 100_000
    data = my_plot_results([log_dir], timesteps, results_plotter.X_EPISODES, "sac1")
    df = xyListToDataFrame(data)
    return df


# In[142]:


sac_df1 = getDataFrame('/mnt/persistent/lane-follow-sac/tmp_1')
sac_df1['Algorithm'] = 'SAC'
sac_df2 = getDataFrame('/mnt/persistent/lane-follow-sac/tmp_2')
sac_df2['Algorithm'] = 'SAC'
sac_df3 = getDataFrame('/mnt/persistent/lane-follow-sac/tmp_3')
sac_df3['Algorithm'] = 'SAC'
sac_df4 = getDataFrame('/mnt/persistent/lane-follow-sac/tmp_4')
sac_df4['Algorithm'] = 'SAC'
sac_df = pd.concat([sac_df1[:100], sac_df2[:100], sac_df3[:100], sac_df4[:100]]).reset_index()


# In[143]:


ppo_df1 = getDataFrame('/mnt/persistent/lane-follow-ppo/tmp_1')
ppo_df1['Algorithm'] = 'PPO'
ppo_df2 = getDataFrame('/mnt/persistent/lane-follow-ppo/tmp_2')
ppo_df2['Algorithm'] = 'PPO'
ppo_df3 = getDataFrame('/mnt/persistent/lane-follow-ppo/tmp_3')
ppo_df3['Algorithm'] = 'PPO'
ppo_df4 = getDataFrame('/mnt/persistent/lane-follow-ppo/tmp_4')
ppo_df4['Algorithm'] = 'PPO'
ppo_df = pd.concat([ppo_df1[:100], ppo_df2[:100], ppo_df3[:100], ppo_df4[:100]]).reset_index()


# In[144]:


a2c_df1 = getDataFrame('/mnt/persistent/lane-follow-a2c/a2c_stacked_gray1')
a2c_df1['Algorithm'] = 'A2C'
a2c_df2 = getDataFrame('/mnt/persistent/lane-follow-a2c/a2c_stacked_gray2')
a2c_df2['Algorithm'] = 'A2C'
a2c_df3 = getDataFrame('/mnt/persistent/lane-follow-a2c/a2c_stacked_gray3')
a2c_df3['Algorithm'] = 'A2C'
a2c_df4 = getDataFrame('/mnt/persistent/lane-follow-a2c/a2c_stacked_gray4')
a2c_df4['Algorithm'] = 'A2C'
a2c_df = pd.concat([a2c_df1[:100], a2c_df2[:100], a2c_df3[:100], a2c_df4[:100]]).reset_index()


# In[145]:


td3_df1 = getDataFrame('/mnt/persistent/lane-follow-td3/td3-trial1-lane-follow')
td3_df1['Algorithm'] = 'TD3'
td3_df2 = getDataFrame('/mnt/persistent/lane-follow-td3/td3-trial2-lane-follow')
td3_df2['Algorithm'] = 'TD3'
td3_df3 = getDataFrame('/mnt/persistent/lane-follow-td3/td3-trial3-lane-follow')
td3_df3['Algorithm'] = 'TD3'
td3_df4 = getDataFrame('/mnt/persistent/lane-follow-td3/td3-trial4-lane-follow')
td3_df4['Algorithm'] = 'TD3'
td3_df = pd.concat([td3_df1[:100], td3_df2[:100], td3_df3[:100], td3_df4[:100]]).reset_index()


# In[152]:


combined_df = pd.concat([sac_df,ppo_df,a2c_df,td3_df])
sns.lineplot(combined_df, x='Episode',y='Reward',hue='Algorithm')


# In[ ]:




