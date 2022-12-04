'''
Created on 4 Dec 2022

@author: lukas
'''

#Built-in imports
from os.path import join

#Third-party imports
import numpy as np

# Local imports
from parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid

from simple_worm_experiments.contraction_relaxation.contraction_relaxation import wrap_contraction_relaxation
from simple_worm_experiments.grid_loader import GridLoader, GridPoolLoader
from mp_progress_logger import FWProgressLogger 


# Iterative solve
# CFL condition dt/dx < C, by making dt 10 times smaller
# while increasing N by a factor 5, we decrease the
# ratio on the l.h.s of the inequality a factor of 2 which hopefully
# helps with stability 
dt_arr = [0.01, 0.001, 0.0001]
N_arr =  [100, 500, 2500]

# Single solve
N = N_arr[0]
dt = dt_arr[0]

data_path = '../../data/contraction_relaxation/'
log_dir = join(data_path, 'logs')
output_dir = join(data_path, 'simulations')

def get_base_parameter():
    
    # Simulation parameter
    N = 100
    dt = 0.01
    T = 1.0
    T0 = 0.5 
    N_report = None
    dt_report = None
    
    # Model parameter
    external_force = ['rft']
    rft = 'Whang'
    use_inertia = False
    
    # Solver parameter
    pi_alpha0 = 0.9
    pi_maxiter = 1000
    fdo = {1: 2, 2:2}
        
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
    smo = True
    a_smo = 150
    
    du_smo = 0.05

    gmo_t = True
    tau_on = 0.05
    Dt_on = 3*tau_on
    tau_off = 0.05 
    Dt_off = 3*tau_off
        
    parameter = {}
              
    parameter['N']  = N
    parameter['dt'] = dt
    parameter['T'] = T
    parameter['T0'] = T0
    
    parameter['N_report'] = N_report
    parameter['dt_report'] = dt_report
        
    parameter['pi_alpha0'] = pi_alpha0            
    parameter['pi_maxiter'] = pi_maxiter             
    parameter['fdo'] = fdo
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft    
        
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
    
    parameter['gmo_t'] = gmo_t
    parameter['tau_on'] = tau_on 
    parameter['Dt_on'] = Dt_on 
    parameter['tau_off'] = tau_on 
    parameter['Dt_off'] = Dt_off 

    return parameter


def simulate_experiments(PG,
                         N_arr,                                              
                         dt_arr,
                         exper_spec,
                         iterative_solve,
                         overwrite):

    '''
    Simulate forward undulation experiments for all points in given ParameterGrid in parallel.
    
    :param PG (ParameterGrid): Parameter grid object
    :param exper_spec (str): String to specify experiment
    :param iterative_solve: If true, solve iteratively with increasing spatial and temporal resolution
    :param overwrite: If true, overwrite already existing files 
    '''
    
    if not iterative_solve:
    
        PGL = FWProgressLogger(PG, 
                               log_dir, 
                               pbar_to_file = False,                        
                               pbar_path = './pbar/pbar.txt', 
                               exper_spec = exper_spec,
                               debug = True)

    if iterative_solve:

        PGL.iterative_run_pool(N_worker, 
                               wrap_contraction_relaxation, 
                               N_arr, 
                               dt_arr,
                               output_dir,
                               overwrite = overwrite)

    else:    
                
        PGL.run_pool(N_worker, 
                     wrap_contraction_relaxation, 
                     output_dir,
                     overwrite = overwrite)


def sim_dt():
    
    parameter = get_base_parameter()
    
    # Make fluid very viscous
    parameter['mu'] = 1e-0

    # Increase simulation time
    parameter['T'] = 4.0
    parameter['T0'] = 2.0
    parameter['dt'] = dt
        
    # Muscle time scale    
    tau_on = 0.05    
    parameter['tau_on']  = tau_on
    parameter['Dt_on'] = 5*tau_on
                
    tau_off = 0.05
                    
    parameter['tau_off'] = tau_off
    parameter['Dt_off'] = 5*tau_off
        
    # Set body viscosities
    parameter['eta'] = 1e-2 * parameter['E'] 
    parameter['nu']  = 1e-2 * parameter['G']
        
    parameter['k0'] = [np.pi, 0, 0]
    parameter['sig0'] = [0, 0, 0]
    parameter['pi'] = False

    dt_param = {'v_min': -2.0, 'v_max': -3.01, 'N': None, 'step': -1, 'round': 3, 'log': True, 'scale': None, 'inverse': False}        

    grid_param = {'dt': dt_param}

    PG = ParameterGrid(parameter, grid_param)
    
    exper_spec = 'CRE'
    
    simulate_experiments(PG,
                         N_arr,                                              
                         dt_arr,
                         exper_spec,
                         iterative_solve = False,
                         overwrite = False)
                                                            
    return

if __name__ == '__main__':
    
    N_worker = 1
    
    sim_dt()
    










