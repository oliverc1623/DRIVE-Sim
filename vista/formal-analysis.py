#!/usr/bin/env python
# coding: utf-8

# In[38]:


from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from typing import Optional
from typing import Tuple
import seaborn as sns
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})


# In[94]:


def my_plot_results(
    dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    return xy_list

def xyListToDataFrame(xylist):
    names = ['Steps', 'Reward']
    df = pd.DataFrame(xylist, columns=names)
    df = df.explode(['Steps','Reward'])
    return df

def getDataFrame(log_dir):
    timesteps = 100_000
    data = my_plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "sac1")
    df = xyListToDataFrame(data)
    return df


# In[113]:


sac_df1 = getDataFrame('sac/tmp_sac_gray1')
sac_df1['Algorithm'] = 'SAC'

sac_df2 = getDataFrame('sac/tmp_sac_gray2')
sac_df2['Algorithm'] = 'SAC'

sac_df3 = getDataFrame('sac/tmp_sac_gray3')
sac_df3['Algorithm'] = 'SAC'

sac_df4 = getDataFrame('sac/tmp_sac_gray4')
sac_df4['Algorithm'] = 'SAC'

sac_df = pd.concat([sac_df1, sac_df2, sac_df3, sac_df4]).reset_index()
sns.lineplot(sac_df,x='Steps',y='Reward',hue='Algorithm',err_style="band",errorbar ='sd',estimator=np.median)


# In[ ]:




