'''
Created on 31 Jul 2022

@author: lukas
'''

from os.path import isdir, expanduser, join
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local imports
from parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid

from simple_worm.plot3d_cosserat import plot_single_strain_vs_control
from simple_worm_experiments.forward_undulation.undulation import wrap_simulate_undulation
from simple_worm_experiments.util import simulate_batch_parallel, write_sim_data_to_file 
from simple_worm_experiments.util import comp_mean_com_velocity, get_filename_from_PG_arr

from mp_progress_logger import FWProgressLogger 

data_path_mju = expanduser('~') + '/mnt/mju/git/forward-worm/data/forward_undulation/'
fig_path = '../../figures/forward_undulation/'


data_path = '../../data/forward_undulation/'
log_dir = join(data_path, 'logs')
output_dir = join(data_path, 'simulations')

# Gazzola, M., Dudte, L. H., McCormick, A. G., & Mahadevan, L. (2018). 
# Forward and inverse problems in the mechanics of soft filaments. 
# Section 3.3 
a = 1

dt_arr = [0.01, 0.01/4, 0.01/4**2, 0.01/4**3]
N_arr =  [100, 250, 500, 1000]

#N_arr  = [int(a/dt) for dt in dt_arr]

dt_0 = dt_arr.pop(0)
N0 = N_arr.pop(0)

def get_base_parameter():
    
    # Simulation parameter
    N = 129
    dt = 0.01
    
    # Model parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
    
    # Solver parameter
    pi_alpha = 0.9
    pi_maxiter = 1000
        
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
                            _try = True,
                            quiet = False)
    
    PG.save(data_path + 'kinematic_parameter/', prefix = 'E_')

    print('Finished!')
    
    return

def get_lam_A_grid():

    base_parameter = get_base_parameter()
    base_parameter['N'] = N0
    base_parameter['T'] = 0.5
    base_parameter['dt'] = dt_0       
    base_parameter['fdo'] = {1: 2, 2: 2}
    base_parameter['pi_alpha0_max'] = 1.0            
    base_parameter['pi_alpha0_min'] = 0.1
    base_parameter['pi_rel_err_growth_tol'] = 4.0
    base_parameter['pi_maxiter_no_progress'] = 50
    base_parameter['pi_tol'] = 1e-5
                                
    lam_param = {'v_min': 0.4, 'v_max': 2.0, 'N': None, 'step': 0.1, 'round': 1, 'log': False, 'scale': None, 'inverse': False}        

    A_param = lam_param.copy()    
    A_param['inverse'] = True

    #c_arr = np.array([np.pi, 2*np.pi, 3*np.pi]) 
    c_arr = np.array([2*np.pi]) 
    #c_arr = np.array([3*np.pi]) 
            
    PG_arr = []
    
    for c in c_arr:

        A_param['scale'] = c

        grid_param = {('lam', 'A'): [lam_param, A_param]}
                        
        PG = ParameterGrid(base_parameter, grid_param)                
        PG_arr.append(PG)

        grid_param = {('lam', 'A'): [lam_param, A_param]}
        
        PG_arr

    return PG_arr, c_arr

def sim_lam_A():
                
    PG_arr, c_arr = get_lam_A_grid()

    exper_spec = 'Forward undulation, simulations for different Lambda and A'
    
    N_worker = 6
    
    for PG, c in zip(PG_arr, c_arr):

        print(f'Simulate c={c:.1f}: #{len(PG)} simulations')
     
        PGL = FWProgressLogger(PG, 
                               log_dir, 
                               pbar_to_file = True,                        
                               pbar_path = join(data_path, 'pbar/pbar.txt'), 
                               exper_spec = exper_spec)

        PGL.iterative_run_pool(N_worker, 
                               wrap_simulate_undulation, 
                               N_arr, 
                               dt_arr,
                               output_dir = join(data_path, 'simulations'),
                               overwrite = True)
                                
    return PG_arr, c_arr

def save_lam_A(PG_arr):
        
    output_keys = ['x', 'times']
                                 
    file_path = write_sim_data_to_file(data_path + 'simulations/', PG_arr, output_keys)

    print(f'Saved file to {file_path}')

    return
    
def plot_lam_A(PG_arr, c_arr):
        
    file_path = data_path + 'simulations/' + get_filename_from_PG_arr(PG_arr) 
    data_dict = pickle.load(open(data_path + file_path, 'rb'))

    ax = plt.subplot(111)
    
    for i, PG in enumerate(PG_arr):
    
        data = data_dict[PG.filename]
        
        x_list = data['x']
        t_list = data['times']

        Delta_T = 1./PG.base_parameter['f']
        
        lam_arr = np.array([v[0] for v in PG.v_arr])
        U_arr = np.zeros(len(x_list))
    
        for j, (x_arr, t_arr) in enumerate(zip(x_list, t_list)):
                            
            U = comp_mean_com_velocity(x_arr, t_arr, Delta_T = Delta_T)
            U_arr[j] = U

        ax.plot(lam_arr, U_arr, 'o-', label = c_arr[i])

    plt.legend()
    plt.show()
      
    return
    
def check_if_pic(PG):
    
    dp = data_path + 'simulations/'

    file_grid = load_file_grid(PG.hash_grid, dp)
    file_arr = file_grid.flatten()
    
    pic_arr = np.concatenate([file['FS'].pic for file in file_arr])
        
    print(f'PI-solver converged at every time step of every simulation: {np.all(pic_arr.flatten())}')
    
    return np.all(pic_arr.flatten())
        
if __name__ == '__main__':
    
    # PG_f = sim_frequency()
    # plot_1d_scan(PG_f, key = 'f', v_arr = PG_f.v_arr, semilogx = False)    
            
    # PG_lam = sim_wavelength()        
    # plot_1d_scan(PG_lam, key = 'lam', v_arr = PG_lam.v_arr, semilogx = False)    


    #PG_lam_A_arr, c_arr = get_lam_A_grid()        
    #print(len(PG_lam_A_arr[0]))
    
    #print([PG.filename for PG in PG_lam_A_arr])        
    P_lam_A_arr, c_arr = sim_lam_A()    
    #save_lam_A(PG_lam_A_arr)
    # load_lam_A(PG_lam_A_arr)
    #plot_lam_A(PG_lam_A_arr, c_arr)
        
    #PG_lam_A_arr, c_arr = get_lam_A_grid()   



    
    

    





