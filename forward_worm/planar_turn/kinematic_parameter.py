'''
Created on 16 Aug 2022

@author: lukas
'''
from experiments.out_of_plan_turn.out_of_plane_turn import base_parameter


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
#from parameter_scan.util import load_file_grid,load_grid_param, 

from simple_worm_experiments.planar_turn.planar_turn import wrap_simulate_planar_turn
from simple_worm_experiments.util import simulate_batch_parallel, get_solver, check_if_pic

# from forward_undulation.plot_undu import plot_1d_scan
# from simple_worm.plot3d_cosserat import plot_single_strain_vs_control
# from simple_worm_experiments.util import comp_mean_com_velocity

data_path = '../../data/planar_turn/'
fig_path = '../../figures/planar_turn/'

def get_base_parameter():
    
    # Simulation base_parameter
    N = 129
    dt = 0.01
    
    # Model base_parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
    
    # Solver base_parameter
    pi_alpha = 0.9
    pi_maxiter = 1000
        
    # Geometric base_parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
    rc = 'spheroid'
    eps_phi = 1e-3
    
    # Material base_parameter
    E = 1e5
    G = E / (2 * (1 + 0.5))
    eta = 1e-2 * E 
    nu = 1e-2 * G
    
    # Fluid 
    mu = 1e-3
    
    # Kinematic base_parameter
    A0 = 4.0
    lam0 = 1.5
    f0 = 2.0    
    
    A1 = 8.0
    lam1 = 1.5
    f1 = 2.0    
       
    T0 = 1.0 / f0
    T1 = 1.0 / f1
       
    T = round(8.0*T0)
    t0 = 4*T0
    Delta_t = 0.25 * T1
        
    smo = True
    a_smo = 150
    du_smo = 0.05
        
    base_parameter = {}
             
    base_parameter['pi_alpha'] = pi_alpha             
    base_parameter['pi_maxiter'] = pi_maxiter             
                
    base_parameter['external_force'] = external_force
    base_parameter['use_inertia'] = use_inertia
    base_parameter['rft'] = rft    
    
    base_parameter['N']  = N
    base_parameter['dt'] = dt
    base_parameter['T'] = T
        
    base_parameter['L0'] = L0
    base_parameter['r_max'] = r_max
    base_parameter['rc'] = rc
    base_parameter['eps_phi'] = eps_phi
    
    base_parameter['E'] = E
    base_parameter['G'] = G
    base_parameter['eta'] = eta
    base_parameter['nu'] = nu
    base_parameter['mu'] = mu

    base_parameter['A0'] = A0
    base_parameter['lam0'] = lam0
    base_parameter['f0'] = f0
        
    base_parameter['A1'] = A1
    base_parameter['lam1'] = lam1
    base_parameter['f1'] = f1    
    
    base_parameter['T'] = T    
    base_parameter['t0'] = t0
    base_parameter['Delta_t'] = Delta_t   
        
        
    base_parameter['smo'] = smo
    base_parameter['a_smo'] = a_smo
    base_parameter['du_smo'] = du_smo
    
    return base_parameter

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

    base_parameter['T'] = 0.001
    base_parameter['N'] = 257
    base_parameter['dt'] = 0.0001
    base_parameter['fdo'] = {1: 2, 2: 2}
    base_parameter['pi_alpha0_max'] = 0.7            
    base_parameter['pi_alpha0_min'] = 0.1
    base_parameter['pi_rel_err_growth_tol'] = 4.0
        
    A0 = base_parameter['A0']  
        
    A1_param = {'v_min': 1.5, 'v_max': 1.51, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': A0}
    
    grid_param = {'A1': A1_param}

    PG = ParameterGrid(base_parameter, grid_param)

    solver = get_solver(base_parameter)
    print(solver)
    
    # N_worker = 1
    #
    # simulate_batch_parallel(N_worker, 
    #                         data_path,                            
    #                         wrap_simulate_planar_turn, 
    #                         PG,                             
    #                         overwrite = False, 
    #                         save = 'all', 
    #                         _try = True)
    #
    # PG.save(data_path + 'kinematic_parameter/', prefix = 'A1_')
    #
    # print('Finished!')
    #
    # return PG
    
def sim_Delta_t():    

    base_parameter = get_base_parameter()
    base_parameter['T'] = 5.0

    T_undu = 1./base_parameter['f']
        
    Delta_t_param = {'v_min': 0.0, 'v_max': 1.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': T_undu}
    
    grid_param = {'Delta_t': Delta_t_param}

    PG = ParameterGrid(base_parameter, grid_param)

    solver = get_solver(base_parameter)
    print(solver)

    # N_worker = 4
    #
    # simulate_batch_parallel(N_worker, 
    #                         data_path,                            
    #                         wrap_simulate_planar_turn, 
    #                         PG,                             
    #                         overwrite = False, 
    #                         save = 'all', 
    #                         _try = True)
    #
    # PG.save(data_path + 'kinematic_parameter/', prefix = 'A1_')
    #
    # print('Finished!')

    return PG

if __name__ == '__main__':
    
    PG_A1 = sim_A1()
        
    #check_if_pic(data_path, PG_A1)
            




    
    

    





