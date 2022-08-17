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
from simple_worm_experiments.util import simulate_batch_parallel, get_solver
from forward_undulation.plot_undu import plot_1d_scan
from simple_worm_experiments.util import comp_mean_com_velocity

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
    A = 4.0
    lam = 1.5
    f = 2.0    
    T = 2.5
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
        
    parameter['A'] = A
    parameter['lam'] = lam
    parameter['f'] = f    
    parameter['smo'] = smo
    parameter['a_smo'] = a_smo
    parameter['du_smo'] = du_smo
    
    return parameter

def sim_frequency():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    f_param = {'v_min': 0.2, 'v_max': 5.01, 'N': None, 'step': 0.2, 'round': 1, 'log': False, 'scale': None}
    
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
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 'E_')

    print('Finished!')

    return PG

def sim_wavelength():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['N'] = 129  
    base_parameter['dt'] = 0.01
    base_parameter['pi_alpha'] = 0.9
    base_parameter['pi_maxiter'] = 500

    lam_param = {'v_min': 2.61, 'v_max': 3.4, 'N': None, 'step': 0.2, 'round': 1, 'log': False, 'scale': None}
    
    grid_param = {'lam': lam_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 'E_')

    print('Finished!')
    
    return

def sim_lam_A():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['N'] = 257  
    base_parameter['dt'] = 0.001
    base_parameter['pi_alpha'] = 0.9
    base_parameter['pi_maxiter'] = 1000
        
    lam_param = {'v_min': 0.4, 'v_max': 2.51, 'N': None, 'step': 0.2, 'round': 1, 'log': False, 'scale': None, 'inverse': False}        
    A_param = lam_param.copy()    
    A_param['inverse'] = True

    #c_arr = np.array([np.pi, 2*np.pi, 3*np.pi]) 
    
    c_arr = np.array([3*np.pi])
    
    PG_arr = []
    
    for c in c_arr:

        A_param['scale'] = 1./c

        grid_param = {('lam', 'A'): [lam_param, A_param]}
                        
        PG = ParameterGrid(base_parameter, grid_param)                
        PG_arr.append(PG)
    
        print(f'Simulate c={c:.1f}: #{len(PG)} simulations')
     
        N_worker = 4

        simulate_batch_parallel(N_worker, 
                                data_path,                            
                                wrap_simulate_undulation, 
                                PG,                             
                                overwrite = False, 
                                save = 'all', 
                                _try = True)
        
        PG.save(data_path + 'kinematic_parameter/', prefix = 'E_')


    return PG_arr, c_arr

def plot_lam_A(PG_arr, c_arr):
    
    dp = data_path + 'simulations/'

    ax = plt.subplot(111)
    
    for i, PG in enumerate(PG_arr):
    
        file_grid = load_file_grid(PG.hash_grid, dp)
        FS_arr = [file['FS'] for file in file_grid]

        Delta_T = 1./PG.base_parameter['f']
        
        lam_arr = np.array([v[0] for v in PG.v_arr])
        U_arr = []
    
        for FS in FS_arr:
                            
            U = comp_mean_com_velocity(FS, Delta_T = Delta_T)
            U_arr.append(U)

        ax.plot(lam_arr, U_arr, 'o-', label = c_arr[i])

    plt.legend()
    plt.show()
      
    return
    
if __name__ == '__main__':
    
    # PG_f = sim_frequency()
    # plot_1d_scan(PG_f, key = 'f', v_arr = PG_f.v_arr, semilogx = False)    
            
    # PG_lam = sim_wavelength()        
    # plot_1d_scan(PG_lam, key = 'lam', v_arr = PG_lam.v_arr, semilogx = False)    

    PG_lam_A_arr, c_arr = sim_lam_A()
    plot_lam_A(PG_lam_A_arr, c_arr)



    
    

    





