'''
Created on 26 Jul 2022

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

def sim_youngs_modulus():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    #base_parameter['gamma'] = 0.1

    E_param = {'v_min': 3.0, 'v_max': 6.0, 'N': 10, 'step': None, 'round': 0, 'log': True, 'scale': None}
    
    phi = 0.3 # Poisson ratio
    c_GE = 1.0/(2*(1 + phi))
    
    G_param = E_param.copy()
    G_param['scale'] = c_GE

    c_etaE = 1e-2    
    eta_param = E_param.copy()
    eta_param['scale'] = c_etaE
    
    c_nuG = 1e-2

    nu_param = E_param.copy()
    nu_param['scale'] = c_GE * c_nuG 

    grid_param = {('E', 'G', 'eta', 'nu'): [E_param, G_param, eta_param, nu_param]}

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

def sim_eta_nu():

    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    E = 1e5
    G = E / (2 * (1 + 0.5))
    base_parameter['E'] = E
    base_parameter['G'] = G
            
    eta_param = {'v_min': -2.5, 'v_max': 0, 'N': None, 'step': 0.1, 'round': 3, 'log': True, 'scale': E}
    nu_param = eta_param.copy()
    nu_param['scale'] = G
    
    grid_param = {('eta', 'nu'): [eta_param, nu_param]}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'material_parameter/', prefix = 'eta_')

    print('Finished!')

    return PG

def plot_youngs_modulus(PG):
    
    dp = data_path + 'simulations/'
    prefix = 'forward_undulation_'

    file_grid = load_file_grid(PG.hash_grid, dp, prefix)

    FS_arr = [file['FS'] for file in file_grid]

    E_arr = [v[0] for v in PG.v_arr]

    plot_1d(PG, key = 'E', v_arr = E_arr, semilogx = True)    
    
    return

def plot_eta_nu(PG):
    
    dp = data_path + 'simulations/'
    prefix = 'forward_undulation_'

    file_grid = load_file_grid(PG.hash_grid, dp, prefix)

    FS_arr = [file['FS'] for file in file_grid]

    eta_arr = np.array([v[0] for v in PG.v_arr])
    E = PG.base_parameter['E']

    c_etaE = eta_arr / E

    plot_1d(PG, key = '\eta/E', v_arr = c_etaE, semilogx = True)    
    
    return

def plot_curvature(PG, key, v_arr, title):

    dp = data_path + 'simulations/'
    prefix = 'forward_undulation_'

    file_grid = load_file_grid(PG.hash_grid, dp, prefix)

    FS_arr = [file['FS'] for file in file_grid]
    CS_arr = [file['CS'] for file in file_grid]
    
    dir_path = fig_path + 'curvature/' + PG.filename 
    
    if not isdir(dir_path):    
        mkdir(dir_path)
    
    
    N = len(FS_arr)
    M = len(str(N))
                 
    for i, (FS, CS, v) in enumerate(zip(FS_arr, CS_arr, v_arr)):
                        
        k = FS.Omega[:, 0, :]
        k0 = CS.Omega[:, 0, :]
                
        plot_single_strain_vs_control(k, k0, dt = PG.base_parameter['dt'], titles=[f'${title}={v}$', None])
        
        j = (M - len(str(i)))*'0' + str(i)
            
        plt.savefig(dir_path  + f'/{j}_{key}={v:.3f}.png')
        
    return

def plot_curvature_eta_nu(PG):
    
    eta_arr = np.array([v[0] for v in PG.v_arr])
    E = PG.base_parameter['E']
    
    v_arr = eta_arr/E    
    title = '\eta/E'
    key = 'c_etaE' 

    plot_curvature(PG, key, v_arr, title)

    return

def plot_curvature_E(PG):
    
    E_arr = np.array([v[0] for v in PG.v_arr])
    
    key = 'E'
    title = 'E'

    plot_curvature(PG, key, E_arr, title)
    
if __name__ == '__main__':
    
    PG_E = youngs_modulus()
    #PG_eta_nu = eta_nu()
    plot_youngs_modulus(PG_E)
    #plot_eta_nu(PG_eta_nu)

    #plot_curvature_eta_nu(PG_eta_nu)
    #plot_curvature_E(PG_E)
    
    




    