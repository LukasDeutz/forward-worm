'''
Created on 27 Jul 2022

@author: lukas
'''
import matplotlib.pyplot as plt

from parameter_scan.util import load_grid_param, load_file_grid
from simple_worm_experiments.forward_undulation.plot_undulation import *
from matplotlib.pyplot import semilogx

data_path = '../../data/forward_undulation/'
fig_path = '../../figures/forward_undulation/'

def plot_1d(PG, key = None, v_arr = None, semilogx = False):
    
    dp = data_path + 'simulations/'
    pref = 'forward_undulation_'

    file_grid = load_file_grid(PG.hash_grid, dp, pref)

    FS_arr = [file['FS'] for file in file_grid]

    if v_arr is None:
        v_arr = [v[0] for v in PG.v_arr]
    if key is None:
        key = PG.keys[0]
                
    gs = plt.GridSpec(1, 4)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
        
    cl = plot_mean_center_of_mass_velocity(ax1, # @UndefinedVariable
                                           FS_arr, 
                                           key, 
                                           v_arr, 
                                           1.0/PG.base_parameter['f'], 
                                           semilogx = semilogx)  
    
    # M = int(len(FS_arr)/2) # N-half    
    # FS_sub_arr = [FS_arr[0], FS_arr[M], FS_arr[-1]]
    # c_sub_l = [cl[0], cl[M], cl[-1]]
        
    plot_center_of_mass_velocity(ax2, FS_arr, color_list = cl)  # @UndefinedVariable
    plot_single_point_trajactory(ax3, FS_arr, point = 'head', undu_plane = 'xy', color_list = cl) # @UndefinedVariable
    plot_center_of_mass_trajectory(ax4, FS_arr, color_list = cl)  # @UndefinedVariable
    
    plt.show()
    
    return

def plot_curvature():
    
    pass
    
