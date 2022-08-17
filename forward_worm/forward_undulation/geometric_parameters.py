'''
Created on 9 Aug 2022

@author: lukas
'''

from os.path import isdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Local imports
from parameter_scan.parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid

from simple_worm_experiments.forward_undulation.undulation import wrap_simulate_undulation
from simple_worm_experiments.util import simulate_batch_parallel, dict_hash
from simple_worm_experiments.plot import plot_comparison_vector_functions_over_time

from plot import plot_nm, plot_curvature

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
    rc = 'spheroid'
            
    # Geometric parameter    
    L0 = 1130 * 1e-6
    r_max = 32 * 1e-6
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
    smo = False
        
    parameter = {}
                
    parameter['external_force'] = external_force
    parameter['use_inertia'] = use_inertia
    parameter['rft'] = rft
    parameter['rc'] = rc
    parameter['eps_phi'] = eps_phi
    
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
    parameter['smo'] = smo
    parameter['a_smo'] = 150
    parameter['du_smo'] = 0.05
        
    return parameter

def sim_a_smo():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['rc'] = 'spheroid'
    #base_parameter['rc'] = 'cylinder'    
    base_parameter['smo'] = True
    base_parameter['du_smo'] = 0.05
    
    a_smo_param = {'v_min': 10.0, 'v_max': 151.0, 'N': None, 'step': 10, 'round': 0, 'log': False, 'scale': None}

    grid_param = {'a_smo': a_smo_param}
    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)
    
    return PG

def sim_phi():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['rc'] = 'spheroid'
    base_parameter['smo'] = True
    base_parameter['a_smo'] = 50
    base_parameter['du_smo'] = 0.05
    
    eps_phi_param = {'v_min': -3, 'v_max': 2, 'N': 6, 'step': None, 'round': 3, 'log': True, 'scale': None}
                            
    grid_param = {'eps_phi': eps_phi_param}

    PG = ParameterGrid(base_parameter, grid_param)

    N_worker = 4

    simulate_batch_parallel(N_worker, 
                            data_path,                            
                            wrap_simulate_undulation, 
                            PG,                             
                            overwrite = False, 
                            save = 'all', 
                            _try = True)

    print('Finished simulations for different eps_phi!')

    return PG
         
def sim_cylinder_spheroid():
    
    base_parameter = get_base_parameter()
    base_parameter['T'] = 2.5
    base_parameter['smo'] = True
        
        
    parameter_spheroid = base_parameter.copy()
    parameter_spheroid['rc'] = 'spheroid'
    parameter_spheroid['eps_phi'] = 1e2
    
    parameter_cylinder = base_parameter.copy()
    parameter_cylinder['rc'] = 'cylinder'
        
    hash_spheroid = dict_hash(parameter_spheroid)    
    hash_cylinder = dict_hash(parameter_cylinder)
            
    wrap_simulate_undulation(parameter_spheroid, 
                             data_path, 
                             hash_spheroid, 
                             overwrite = False, 
                             save = 'all', 
                             _try = False)
    
    wrap_simulate_undulation(parameter_cylinder, 
                             data_path, 
                             hash_cylinder, 
                             overwrite = False, 
                             save = 'all', 
                             _try = False)
        
    print('Finished!')
    
    return hash_spheroid, hash_cylinder 

def plot_cylinder_spheroid(hash_spheroid, hash_cylinder):
    
    fps = data_path + 'simulations/' + 'forward_undulation_' + hash_spheroid + '.dat'
    fpc = data_path + 'simulations/' + 'forward_undulation_' + hash_cylinder + '.dat'

    data_s = pickle.load(open(fps, 'rb'))
    data_c = pickle.load(open(fpc, 'rb'))

    FS_s = data_s['FS']
    FS_c = data_c['FS']
    
    fds = fig_path + '/force_and_torque/' + hash_spheroid + hash_cylinder

    if not isdir(fds): mkdir(fds)
        
    n_arr = [FS_s.n, FS_c.n]
    m_arr = [FS_s.m, FS_c.m]
     
    nm_list = [n_arr, m_arr]
                      
    color_list = ['b', 'r']

    dir_nm = fds + '/nm/'    
    
    if not isdir(dir_nm): mkdir(dir_nm)
                          
    plot_comparison_vector_functions_over_time(nm_list, dir_nm, color_list, eps = 1e-6)

    print('Finished plotting!')    
        
    return
        
#------------------------------------------------------------------------------ 
#
        
if __name__ == '__main__':
    
    #hs, hc = sim_cylinder_spheroid()    
    #print(hs+hc)    
    #plot_cylinder_spheroid(hs, hc)    
    
    PG_phi = sim_phi()        
    print(PG_phi.filename)
    #plot_nm(PG_phi, PG_phi.v_arr, 'forward_undulation/', log = True)    
    #plot_curvature(PG_phi, 'forward_undulation/')
    
    #PG_a_smo = sim_a_smo()
    #print(PG_a_smo.filename) 
    #plot_nm(PG_a_smo, PG_a_smo.v_arr, 'forward_undulation/', log = False)    
    #plot_curvature(PG_a_smo, 'forward_undulation/')


    
    
    
    


