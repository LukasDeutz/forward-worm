'''
Created on 27 Jul 2022

@author: lukas
'''

from os.path import isdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from parameter_scan.parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid

from simple_worm.plot3d_cosserat import plot_single_strain_vs_control,\
    plot_single_strain

from simple_worm_experiments.forward_undulation.undulation import wrap_simulate_undulation
from simple_worm_experiments.forward_undulation.plot_undulation import *
from simple_worm_experiments.util import simulate_batch_parallel, frame_trafo
from simple_worm_experiments.plot import plot_vector_functions_over_time, plot_comparison_vector_functions_over_time
from plot import plot_curvature

from simple_worm.util_experiments import color_list_from_values

from forward_undulation.plot import plot_1d_scan


data_path = '../../data/forward_undulation/'
fig_path = '../../figures/forward_undulation/'

def get_base_parameter():
    
    # Simulation parameter
    N = 129
    dt = 0.01
    
    # Model parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
        
    # Geometric parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
    rc = 'spheroid'
    
    # Material parameter
    E = 1e5
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G
    
    # Fluid 
    mu = 1e-3
    
    # Kinematic parameter
    A = 4.0
    lam = 1.5
    f = 2.0    
    T = 2.5
    smo = False
        
    parameter = {}
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft
    
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
        
    parameter['L0'] = L0
    parameter['r_max'] = r_max
    parameter['rc'] = rc
    
    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['mu'] = mu
        
    parameter['A'] = A
    parameter['lam'] = lam
    parameter['f'] = f    
    parameter['smooth_muscle_onset'] = smo
    
    return parameter

def sim_c_t():

    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    K = 1.6

    c_t_param = {'v_min': 0.5, 'v_max': 10.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': None}
    c_n_param = c_t_param.copy()    
    c_n_param['scale'] = K
                        
    grid_param = {('c_t', 'c_n'): [c_t_param, c_n_param]}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'rft/', prefix = 'c_t_')

    print('Finished simulations for different c_t!')

    return PG
      
def sim_K():

    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    c_t = 1.0 

    base_parameter['c_t'] = c_t

    c_n_param = {'v_min': 1.0, 'v_max': 10.0, 'N': None, 'step': 0.2, 'round': 1, 'log': False, 'scale': c_t}
                            
    grid_param = {'c_n': c_n_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 1

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'rft/', prefix = 'K_')

    print('Finished simulations for different K!')


    return PG
    
def sim_gamma():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['rc'] = 'spheroid'

    gamma_param = {'v_min': -3, 'v_max': 1.0, 'N': None, 'step': 0.2, 'round': 3, 'log': True, 'scale': None}
                            
    grid_param = {'gamma': gamma_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)

    print('Finished simulations for different gamma!')

    return PG

def plot_mn(PG, v_arr, log = True):
    
    fd = fig_path + '/force_and_torque/' + PG.filename

    if not isdir(fd):
        mkdir(fd)
            
    dp = data_path + 'simulations/'
    pref = 'forward_undulation_'
    
    file_grid = load_file_grid(PG.hash_grid, dp, pref)
                
    FS_arr = [file['FS'] for file in file_grid]

    color_list = color_list_from_values(v_arr, log = log)

    m_arr = [FS.m for FS in FS_arr]
    n_arr = [FS.n for FS in FS_arr]    

    nm_list = [n_arr, m_arr] 

    dir_fnml = fd + '/nm/'    
    if not isdir(dir_fnml): mkdir(dir_fnml)
                      
    plot_comparison_vector_functions_over_time(nm_list, dir_fnml, color_list, eps = 1e-6)
          
    print('Finished plotting!')    

    return


def plot_fnlm(PG, v_arr, log = True):

    fd = fig_path + '/force_and_torque/' + PG.filename

    if not isdir(fd):
        mkdir(fd)
            
    dp = data_path + 'simulations/'
    pref = 'forward_undulation_'
    
    file_grid = load_file_grid(PG.hash_grid, dp, pref)
                
    FS_arr = [file['FS'] for file in file_grid]

    color_list = color_list_from_values(v_arr, log = log)

    f_arr = [FS.f for FS in FS_arr]
    n_arr = [FS.n for FS in FS_arr]    
    l_arr = [FS.l for FS in FS_arr]
    m_arr = [FS.m for FS in FS_arr]

    fnlm_list = [f_arr, n_arr, l_arr, m_arr] 

    dir_fnml = fd + '/fml/'    
    if not isdir(dir_fnml): mkdir(dir_fnml)
                      
    plot_comparison_vector_functions_over_time(fnlm_list, dir_fnml, color_list, eps = 1e-6)
          
    print('Finished plotting!')
                        
    return

def check_linear_balance(PG):
    
    dp = data_path + 'simulations/'
    pref = 'forward_undulation_'
    
    file_grid = load_file_grid(PG.hash_grid, dp, pref)    

    FS_arr = [file['FS'] for file in file_grid]

    for FS in FS_arr:
        
        f_bar = FS.f
        n_bar = FS.n
        
        f = frame_trafo(f_bar, FS, 'b2l')
        n = frame_trafo(n_bar, FS, 'b2l')
            
        N = np.size(f, 2)    
        ds = 1.0/(N-1)
        
        lb = f - np.gradient(n, ds, axis = 2)
    
        #assert np.all(lb <= 1e-3)
    
    return
            
if __name__ == '__main__':

    PG = sim_gamma()
            
    #PG = sim_c_t()    
    #PG_K = sim_K()    
    #plot_K(PG_K)    
    #plot_gamma(PG)
    #plot_curvature(PG)
    
    #plot_1d_scan(PG, 'gamma', PG.v_arr, semilogx = True)    
    #plot_fnlm(PG, PG.v_arr, log = True)
    
    #check_linear_balance(PG)
    
    plot_mn(PG, PG.v_arr, log = True)
    
    #plot_gamma_f_and_l(PG, PG.v_arr, log = True)
    #plot_gamma_f_and_l(PG)
    
    print('Finished')
    
    
    







