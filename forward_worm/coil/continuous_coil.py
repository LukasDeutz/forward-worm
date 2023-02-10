'''
Created on 10 Feb 2023

@author: lukas
'''

#Built-in imports
from os.path import join

#Third-party imports
import numpy as np
import h5py

# Local imports
from forward_worm.coil.coil_dirs import log_dir, sim_dir, sweep_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5, sim_exp_wrapper

from simple_worm_experiments.coiling.coiling import CoilingExperiment
from simple_worm_experiments.grid_loader import GridLoader, GridPoolLoader
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.model_parameter import default_model_parameter

from parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid
from mp_progress_logger import FWProgressLogger 

#------------------------------------------------------------------------------ 
# Parameter

def continuous_coil_parameter():
    
    parameter = default_model_parameter()

    # Gradual muscle onset on the head and the tale        
    parameter['gmo'] = True    
    parameter['Ds_h'] = 0.01
    parameter['s0_h'] = 3*parameter['Ds_h']    
    parameter['Ds_t'] = 0.01
    parameter['s0_t'] = 1 - 3*parameter['Ds_t']

    # Kinematic parameter
    parameter['A_dv'] = 4.0
    parameter['f_dv'] = 5.0    
    parameter['lam_dv'] = 1.5

    parameter['A_lr'] = 4.0
    parameter['f_lr'] = 5.0    
    parameter['lam_lr'] = 1.5
    
    parameter['phi'] = np.pi/2

    parameter['Ds_dv_h'] = None
    parameter['s0_dv_h'] = None
    parameter['Ds_dv_t'] = None
    parameter['s0_dv_t'] = None
    
    parameter['Ds_lr_h'] = None
    parameter['s0_lr_h'] = None
    parameter['Ds_lr_t'] = 1.0/32
    parameter['s0_lr_t'] = 0.25

#------------------------------------------------------------------------------ 
# Post processing

def compute_roll_frequency_from_euler_angle(h5, PG):
    '''
    Computes the average roll frequency of the worm's body
    using the time evolution of body averaged roll angle     
    '''

    # roll angle
    alpha_sweep = h5['FS']['theta'][:, :, 0, :]            
    t = h5['times'][:]

    s_arr = np.linspace(0, 1, PG.base_parameter['N'])
    s_mask = s_arr >= PG.base_parameter['s0_lr_t']
                    
    f_R_avg_arr = np.zeros(len(alpha_sweep))
    f_R_std_arr = np.zeros(len(alpha_sweep))
                        
    for i, (alpha, param) in enumerate(zip(alpha_sweep, PG.param_arr)):
                    
        T_undu = 1.0 / param['f_dv']
                            
        f_avg, f_std = EPP.comp_roll_frequency_from_euler_angle(
            alpha, t, s_mask, Dt = T_undu)
        
        f_R_avg_arr[i] = f_avg
        f_R_std_arr[i] = f_std
    
    if 'f_R_avg' in h5: h5['f_R_avg'][:] = f_R_avg_arr    
    else: h5.create_dataset('f_R_avg', data = f_R_avg_arr)    
    if 'f_R_std' in h5: h5['f_R_std'][:] = f_R_std_arr    
    else: h5.create_dataset('f_R_std', data = f_R_std_arr)
    
    return
            
def sim_coil_A(N_worker,
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):

    '''
    Parameter sweep over curvature amplitude
    '''
    
    param = continuous_coil_parameter()    
    param['T'] = 20.0
    
    # Parameter Grid    
    A_min, A_max, A_step = 1.0, 8.01, 1.0
                                                      
    A_dv_param = {'v_min': A_min, 'v_max': A_max + 0.1*A_step, 'N': None, 'step': A_step, 'round': 1}        
    A_lr_param = A_dv_param.copy()    
                                            
    grid_param = {('A_dv', 'A_lr', 'T') : (A_dv_param, A_lr_param)}
    
    PG = ParameterGrid(param, grid_param)
        
    filename = (f'continuous_coil_A_min={A_min}_A_max={A_max}_A_step={A_step}_'
        f'f={param["f_dv"]}_s0_lr={param["s0_lr_t"]}_'
        f'N={param["N"]}_dt={param["dt"]}.h5')        
    
    h5_filepath = sweep_dir / filename
    exper_spec = 'CCE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        CoilingExperiment.continuous_coiling,
        h5_filepath,
        log_dir,
        sim_dir,
        exper_spec,
        overwrite,
        debug)

    # Analyse raw data and save results
    compute_roll_frequency_from_euler_angle(h5, PG)
    
    return

def sim_coil_f():
    
    pass

def sim_coil_s0_t():
    
    pass

def sim_E():
    
    pass

def sim_E_p():
    
    pass

def sim_mu():
    
    pass

    







    
    
    
    
    
    

