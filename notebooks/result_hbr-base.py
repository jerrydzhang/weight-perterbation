#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import utils.visualize as viz
from utils.ode import map_equation, hydrogen_bromine, hydrogen_bromine_init


# In[2]:


dt = 0.002
visualize = viz.ODEResultVisualizer(
    fun=hydrogen_bromine,
    x0=hydrogen_bromine_init,
    t=np.arange(0, 1e1, dt),
    result_dir="../experiments/results/hbr_20251006-202655",
    labels=["Br2", "Br", "H2", "H", "HBr", "M"],
)
visualize.solve()


# In[3]:


visualize.plot_solution()


# In[4]:


visualize.plot_metric_distribution("nmse_mean")


# In[5]:


visualize.plot_parameter_heatmap(
    param_x="model_alpha",
    param_y="weight_eps",
    metric="nmse_mean",
    fixed_params={"weight_std": 0},
    save=True,
)


# In[6]:


df = visualize.results_df
df.to_csv("temp_data.csv", index=False)


# In[7]:


visualize.get_best_model("nmse_mean", fixed_params={"weight_std": 0})


# In[ ]:


# visualize.plot_best_model_solution('nmse_mean', fixed_params={'weight_std': 0})
model = "21_model"
visualize.plot_model_solution(model, save=True)


# In[ ]:


# visualize.plot_best_model_coefficients('nmse_mean', fixed_params={'weight_std': 0})
visualize.plot_coefficients(model, save=True)


# In[ ]:


# In[ ]:
