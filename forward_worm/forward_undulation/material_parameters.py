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

def sim_E():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    E_param = {'v_min': 0.0, 'v_max': 7.01, 'N': None, 'step': 0.2, 'round': 1, 'log': True, 'scale': None}
    
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

def sim_E_mu():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    E_param = {'v_min': 0.0, 'v_max': 6.0, 'N': None, 'step': 0.2, 'round': 1, 'log': True, 'scale': None}
    
    phi = 0.5 # Poisson ratio
    c_GE = 1.0/(2*(1 + phi))
    
    G_param = E_param.copy()
    G_param['scale'] = c_GE

    c_etaE = 1e-2    
    eta_param = E_param.copy()
    eta_param['scale'] = c_etaE
    
    c_nuG = 1e-2

    nu_param = E_param.copy()
    nu_param['scale'] = c_GE * c_nuG 

    mu_param = {'v_min': -3, 'v_max': -1, 'N': 5, 'step': None, 'round': 3, 'log': True, 'scale': None}
       
    grid_param = {('E', 'G', 'eta', 'nu'): [E_param, G_param, eta_param, nu_param],
                  'mu': mu_param}


    PG = ParameterGrid(base_parameter, grid_param)

    print(f'Simulations to run: {len(PG)}')

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'material_parameter/', prefix = 'E_mu_')

    print('Finished!')

    return PG

def sim_eta():

    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    E = 1e5
    phi = 0.3 # Poisson ratio
    G = E / (2 * (1 + phi))
    
    base_parameter['E'] = E
    base_parameter['G'] = G

    step = 0.2
    v_min = -3.0
    v_cen = -2.0
    v_max = 1.01
            
    #eta_param = {'v_min': -2.0, 'v_max': 1.0, 'N': None, 'step': 0.2, 'round': 3, 'log': True, 'scale': E}
    eta_param = {'v_min': v_min, 'v_max': v_max, 'N': None, 'step': step, 'round': 3, 'log': True, 'scale': E}        
    
    nu_param = eta_param.copy()
    nu_param['scale'] = G
        
    # smaller time resolution    
    N1 = int((v_cen - v_min)/step)
    N2 = int((v_max - v_cen)/step) + 1    
    
    dt_param = {'v_tup': (1e-3, 1e-2), 'N_tup': (N1, N2), 'round': 3, 'log': False, 'scale': None}
    
    grid_param = {('eta', 'nu', 'dt'): [eta_param, nu_param, dt_param]}

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

def sim_eta_mu():

    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5

    E = 1e5
    phi = 0.3 # Poisson ratio
    G = E / (2 * (1 + phi))
    
    base_parameter['E'] = E
    base_parameter['G'] = G

    eta_param = {'v_min': -2.0, 'v_max': 1.01, 'N': None, 'step': 0.2, 'round': 3, 'log': True, 'scale': E}        
    
    nu_param = eta_param.copy()
    nu_param['scale'] = G

    mu_param = {'v_min': -3, 'v_max': -1, 'N': 5, 'step': None, 'round': 3, 'log': True, 'scale': None}
       
    grid_param = {('eta', 'nu'): (eta_param, nu_param),
                  'mu': mu_param}


    PG = ParameterGrid(base_parameter, grid_param)

    print(f'Simulations to run: {len(PG)}')

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    PG.save(data_path + 'material_parameter/', prefix = 'eta_mu_')

    print('Finished!')

    return PG

def plot_x_mu(PG):
    
    dp = data_path + 'simulations/'
             
    x_arr = np.array([v[0] for v in PG.v_arr[0]])
    mu_arr = PG.v_arr[1]

    Delta_T = 1./PG.base_parameter['f']

    fig = plt.figure()
    ax = plt.subplot(111)
            
    for i, mu in enumerate(mu_arr):
        
        _, _, _, hash_arr = PG[:, i]
        
        file_grid = load_file_grid(hash_arr, dp)
        FS_arr = [file['FS'] for file in file_grid]
        
        U_arr = np.zeros(len(FS_arr))
        
        for j, FS in enumerate(FS_arr):
        
            U = comp_mean_com_velocity(FS, Delta_T = Delta_T)
            U_arr[j] = U

        plt.semilogx(x_arr, U_arr, label = f'$\mu={mu}$')
    
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
    
    # PG_E = sim_E()
    # check_if_pic(PG_E)        
    # plot_1d_scan(PG_E, key = 'E', v_arr = [v[0] for v in PG_E.v_arr], semilogx = True)    
        
    # PG_eta = sim_eta()
    # check_if_pic(PG_eta)    
    # eta_E_arr = np.array([v[0] for v in PG_eta.v_arr]) / PG_eta.base_parameter['E']    
    # plot_1d_scan(PG_eta, key = '$\eta$', v_arr = eta_E_arr, semilogx = True)    
       
       
    PG_eta_mu = sim_eta_mu()
    check_if_pic(PG_eta_mu)        
    plot_x_mu(PG_eta_mu)
                
    #PG_E_mu = sim_E_mu()                
    #plot_E_mu(PG_E_mu)
    
    #plot_curvature_eta_nu(PG_eta_nu)
    #plot_curvature_E(PG_E)
    
    




    