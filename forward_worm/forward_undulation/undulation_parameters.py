'''
Created on 31 Jul 2022

@author: lukas
'''

from os.path import isdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from parameter_scan.parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid

from simple_worm.plot3d_cosserat import plot_single_strain_vs_control
from simple_worm_experiments.forward_undulation.undulation import wrap_simulate_undulation
from simple_worm_experiments.util import simulate_batch_parallel

from plot import plot_1d

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

def frequency():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    #base_parameter['gamma'] = 0.1

    f_param = {'v_min': 0.1, 'v_max': 3.0, 'N': 0, 'step': 0.1, 'round': 1, 'log': False, 'scale': None}
    
    grid_param = {'f': f_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'material_parameter/', prefix = 'E_')

    print('Finished!')

    return PG





