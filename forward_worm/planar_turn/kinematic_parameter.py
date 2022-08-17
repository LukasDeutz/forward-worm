'''
Created on 16 Aug 2022

@author: lukas
'''


'''
Created on 31 Jul 2022

@author: lukas
'''
# from os.path import isdir
# from os import mkdir
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from parameter_scan.parameter_scan import ParameterGrid
from parameter_scan.util import load_file_grid#,load_grid_param, 

from simple_worm_experiments.planar_turn.planar_turn import wrap_simulate_planar_turn
from simple_worm_experiments.util import simulate_batch_parallel, get_solver

# from forward_undulation.plot_undu import plot_1d_scan
# from simple_worm.plot3d_cosserat import plot_single_strain_vs_control
# from simple_worm_experiments.util import comp_mean_com_velocity

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
    
    # Solver parameter
    pi_alpha = 0.5
    pi_maxiter = 100
        
    # Geometric parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
    rc = 'spheroid'
    eps_phi = 1e-3
    
    # Material parameter
    E = 1e5
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G
    
    # Fluid 
    mu = 1e-3
    
    # Kinematic parameter
    A0 = 4.0
    lam0 = 1.5
    f0 = 2.0    
    
    A1 = 8.0
    lam1 = 1.5
    f1 = 2.0    
       
    T = int(10.0/f0)
        
    smo = True
    a_smo = 150
    du_smo = 0.05
        
    parameter = {}
             
    parameter['pi_alpha'] = pi_alpha             
    parameter['pi_maxiter'] = pi_maxiter             
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft    
    
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
        
    parameter['L0'] = L0
    parameter['r_max'] = r_max
    parameter['rc'] = rc
    parameter['eps_phi'] = eps_phi
    
    parameter['E'] = E
    parameter['G'] = G
    parameter['eta'] = eta
    parameter['nu'] = nu
    parameter['mu'] = mu

    parameter['A0'] = A0
    parameter['lam0'] = lam0
    parameter['f0'] = f0
        
    parameter['A1'] = A1
    parameter['lam1'] = lam1
    parameter['f1'] = f1    
    
    parameter['T'] = T    
        
    parameter['smo'] = smo
    parameter['a_smo'] = a_smo
    parameter['du_smo'] = du_smo
    
    return parameter

def sim_t0():
    
    base_parameter = get_base_parameter()

    base_parameter['T'] = 5.0
    
    T_undu = 1./base_parameter['f']
        
    t0_param = {'v_min': 0.0, 'v_max': 1.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': T_undu}
    
    grid_param = {'t0': t0_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_planar_turn, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 't0_')

    print('Finished!')

    return PG

def sim_A1():
    
    base_parameter = get_base_parameter()

    base_parameter['T'] = 5.0
    A0 = base_parameter['A0']  
        
    A1_param = {'v_min': 1.0, 'v_max': 2.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': A0}
    
    grid_param = {'A1': A1_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_planar_turn, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 'A1_')

    print('Finished!')

    return PG
    
def sim_Delta_t():    

    base_parameter = get_base_parameter()
    base_parameter['T'] = 5.0

    T_undu = 1./base_parameter['f']
        
    Delta_t_param = {'v_min': 0.0, 'v_max': 1.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': T_undu}
    
    grid_param = {'Delta_t': Delta_t_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_planar_turn, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 'A1_')

    print('Finished!')

    return PG

if __name__ == '__main__':
    
    # PG_f = sim_frequency()
    # plot_1d_scan(PG_f, key = 'f', v_arr = PG_f.v_arr, semilogx = False)    
            
    # PG_lam = sim_wavelength()        
    # plot_1d_scan(PG_lam, key = 'lam', v_arr = PG_lam.v_arr, semilogx = False)    




    
    

    





