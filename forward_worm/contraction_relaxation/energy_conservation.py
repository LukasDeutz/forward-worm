'''
Created on 4 Dec 2022

@author: lukas
'''
#Third-party imports
import numpy as np

# Local imports
from forward_worm.contraction_relaxation.contraction_relaxation_dirs import log_dir, sim_dir, sweep_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5, sim_exp_wrapper

from simple_worm_experiments.contraction_relaxation.contraction_relaxation import ContractionRelaxationExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm_experiments.model_parameter import default_model_parameter

from parameter_scan import ParameterGrid

FS_KEYS = ['Omega', 'sigma', 'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig',
     'dot_W_F_lin', 'dot_W_F_rot', 'dot_W_M_lin', 'dot_W_M_rot']

CS_KEYS = ['Omega', 'sigma']

#------------------------------------------------------------------------------ 
# Parameter

def contraction_relaxation_parameter():
    
    param = default_model_parameter()  

    param['T'] = 6.0
    
    # Muscles switch on and off at finite timescale
    param['fmts'] = True
    param['tau_on'] = 0.03
    param['t_on'] = 5 * param['tau_on']

    param['tau_off'] = 0.03
    param['t_off'] = 3.0 

    # Gradual muscle onset at head and tale
    param['gmo'] = True
    param['Ds_h'] = 0.01
    param['s0_h'] = 3*param['Ds_h']    
    param['Ds_t'] = 0.01
    param['s0_t'] = 1 - 3*param['Ds_t']
                                
    # Make fluid very viscous
    param['mu'] = 1e-0
  
    # Constant curvature          
    param['k0'] = np.pi

    return param

#------------------------------------------------------------------------------ 
# Post processing

def compute_power_out(h5):
    '''
    Compute total power consumption
    
    :param h5:
    '''
                        
    powers = EPP.powers_from_h5(h5, t_start = None, t_end = None)
    
    dot_V = powers['dot_V_k'] + powers['dot_V_sig']
    dot_D = powers['dot_D_k'] + powers['dot_D_sig']
    dot_W_F = powers['dot_W_F_F'] + powers['dot_W_F_T']
    dot_W_M = powers['dot_W_M_F'] + powers['dot_W_M_T']

    dot_E_out = - dot_V + dot_D + dot_W_F
                
    if not 'powers' in h5:  
        grp = h5.create_group('powers')        
        grp.create_dataset('dot_E_out', data = dot_E_out)
        grp.create_dataset('dot_W_M', data = dot_W_M)
    else: 
        grp = h5['powers']
        grp['dot_E_out'][:] = dot_E_out
        grp['dot_W_M'][:] = dot_W_M
                                        
    return
    
#------------------------------------------------------------------------------ 
# Simulations

def sim_dt(N_worker,
       simulate = True,
       save_raw_data = True,
       overwrite = False,
       debug = False):
    
    param = contraction_relaxation_parameter()

    param['pi'] = False
    # Make fluid very viscous
    param['mu'] = 1e-0

    param['N'] = 125
    param['N_report'] = 125

    # Set simulation time and muscle time scale    
    param['tau_on'] = 0.1
    param['tau_off'] = 0.1
    param['t_on'] = 5*param['tau_on']            
    param['t_off'] = 3.0        
    param['T']  = 5.0

    # Parameter Grid        
    #dt_arr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    dt_arr = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    dt_param = {'v_arr': dt_arr, 'round': 5}        
    
    grid_param = {'dt' : dt_param}

    PG = ParameterGrid(param, grid_param)

    filename = (f'contraction_relaxation_dt={1e-4}_dt_max={1e-2}_'
        f'N={param["N"]}_tau_on={param["tau_on"]:.2f}_'
        f'k0={param["k0"]:.2f}_pi={param["pi"]}.h5')        

    h5_filepath = sweep_dir / filename
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,                  
        FS_KEYS,
        CS_KEYS,        
        N_worker,
        PG,
        ContractionRelaxationExperiment.relaxation_control_sequence,
        h5_filepath, log_dir, sim_dir,
        exper_spec, overwrite, debug)
    
    compute_power_out(h5)
    
    return

def sim_N(N_worker,
       simulate = True,
       save_raw_data = True,
       overwrite = False,
       debug = False):
    
    param = contraction_relaxation_parameter()

    # Make fluid very viscous
    param['mu'] = 1e-0

    # Set simulation time and muscle time scale    
    param['tau_on'] = 0.1
    param['tau_off'] = 0.1
    param['t_on'] = 5*param['tau_on']            
    param['t_off'] = 3.0        
    param['T']  = 5.0
        
    param['dt'] = 1e-4

    # Parameter Grid        
    N_arr = [125, 250, 500, 1000, 2000]
    param['N_report'] = N_arr[0]
    
    N_param = {'v_arr': N_arr, 'round': None}            
    grid_param = {'N' : N_param}

    PG = ParameterGrid(param, grid_param)

    filename = (f'contraction_relaxation_N_min={N_arr[0]}_N_max={N_arr[-1]}_'
        f'dt={param["dt"]}_tau_on={param["tau_on"]:.2f}_'
        f'k0={param["k0"]:.2f}_pi={param["pi"]}.h5')        

    h5_filepath = sweep_dir / filename
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,                  
        FS_KEYS,
        CS_KEYS,        
        N_worker,
        PG,
        ContractionRelaxationExperiment.relaxation_control_sequence,
        h5_filepath, log_dir, sim_dir,
        exper_spec, overwrite, debug)
    
    compute_power_out(h5)
    
    return

def sim_tau_on_off(N_worker,
       simulate = True,
       save_raw_data = True,
       overwrite = False,
       debug = False):
    
    param = contraction_relaxation_parameter()

    # Make fluid very viscous
    param['mu'] = 1e-0

    # Increase simulation time            
    param['T'] = 10.0
                
    tau_on_min = 0.01
    tau_on_max = 0.20
    tau_on_step = 0.01

    param['t_on'] = 5*tau_on_max
    param['t_off'] = 5.0

    # Choose time step to resolve smallest muscle time scale
    param['dt'] = 1e-3

    tau_on_param = {'v_min': tau_on_min, 'v_max': tau_on_max + 0.1*tau_on_step, 
        'N': None, 'step': tau_on_step, 'round': 2}                                                

    tau_off_param = tau_on_param.copy()
    grid_param = {('tau_on', 'tau_off') : (tau_on_param, tau_off_param)}
    PG = ParameterGrid(param, grid_param)

    filename = (f'contraction_relaxation_tau_min={tau_on_min}_tau_max={tau_on_max}_'
        f'tau_step={tau_on_step}_k0={param["k0"]:.2f}_mu={param["mu"]}_'
        f'N={param["N"]}_dt={param["dt"]}.h5')        

    h5_filepath = sweep_dir / filename
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,                  
        FS_KEYS,
        CS_KEYS,        
        N_worker,
        PG,
        ContractionRelaxationExperiment.relaxation_control_sequence,
        h5_filepath, log_dir, sim_dir,
        exper_spec, overwrite, debug)
    
    compute_power_out(h5)

    return

# def sim_eta_nu(N_worker,
#         simulate = True,
#         save_raw_data = True,
#         overwrite = False,
#         debug = False):
#
#     param = contraction_relaxation_parameter()
#
#
#     # Make fluid very viscous
#     param['mu'] = 1e-0
#
#     # Increase simulation time
#     param['T']  = 4.0
#     param['T0'] = 2.0
#     param['dt'] = 0.0001
#
#     E = param['E']
#     G = param['G']
#
#     eta_param = {'v_min': -4, 'v_max': 0, 'N': None, 'step': 1.0, 'round': 3, 'log': True, 'scale': E}        
#     nu_param  = {'v_min': -4, 'v_max': 0, 'N': None, 'step': 1.0, 'round': 3, 'log': True, 'scale': G}        
#
#     grid_param = {('eta', 'nu'): [eta_param, nu_param]}
#
#     PG = ParameterGrid(param, grid_param)
#
#     PGL = FWProgressLogger(PG, 
#                            log_dir, 
#                            pbar_to_file = False,                        
#                            pbar_path = './pbar/pbar.txt', 
#                            exper_spec = 'CRE',
#                            debug = True)
#
#     PGL.run_pool(N_worker, 
#                  wrap_contraction_relaxation, 
#                  output_dir,
#                  overwrite = False)
#
#
#     h5_filepath = join(data_path, 'parameter_scans', f'contraction_relaxation_eta_nu.h5')
#
#     FS_keys = ['Omega', 'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 'dot_W_F_lin', 'dot_W_F_rot', 'dot_W_M_lin', 'dot_W_M_rot']
#
#     save_experiment_to_h5(PG, 
#                           h5_filepath,
#                           output_dir,
#                           log_dir, 
#                           FS_keys = FS_keys, 
#                           CS_keys = 'Omega')
#
#     PGL.close()    
#
#     return
#
# def sim_tau_on_tau_off():
#
#     param = get_base_parameter()
#
#     # Make fluid very viscous
#     param['mu'] = 1e-0
#
#     # Increase simulation time
#     param['T']  = 4.0
#     param['T0'] = 2.0
#     param['dt'] = 0.001
#     param['dt_report']
#
#     tau_on_param = {'v_min': 0.01, 'v_max': 0.11, 'N': None, 'step': 0.01, 'round': 2}            
#     Dt_on_param = tau_on_param.copy()
#     Dt_on_param['scale'] = 5
#
#     tau_off_param = tau_on_param.copy()
#     Dt_off_param = Dt_on_param.copy()
#
#     grid_param = {('tau_on', 'Dt_on', 'tau_off', 'Dt_off'): [tau_on_param, Dt_on_param, tau_off_param, Dt_off_param]}
#
#     PG = ParameterGrid(param, grid_param)
#
#     PGL = FWProgressLogger(PG, 
#                            log_dir, 
#                            pbar_to_file = False,                        
#                            pbar_path = './pbar/pbar.txt', 
#                            exper_spec = 'CRE',
#                            debug = False)
#
#     PGL.run_pool(N_worker, 
#                  wrap_contraction_relaxation, 
#                  output_dir,
#                  overwrite = False)
#
#
#     h5_filepath = join(data_path, 'parameter_scans', f'contraction_relaxation_tau_on_tau_off.h5')
#
#     FS_keys = ['Omega', 'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 'dot_W_F_lin', 'dot_W_F_rot', 'dot_W_M_lin', 'dot_W_M_rot']
#
#     save_experiment_to_h5(PG, 
#                           h5_filepath,
#                           output_dir,
#                           log_dir, 
#                           FS_keys = FS_keys, 
#                           CS_keys = 'Omega')
#
#     PGL.close()    
#
#     return

if __name__ == '__main__':
    
    N_worker = 5
    
    sim_dt(N_worker, simulate = False, 
        save_raw_data = True, overwrite = False, debug = False)

    sim_N(N_worker, simulate = False,
        save_raw_data = True, overwrite = False, debug = False)

    # sim_tau_on_off(N_worker, simulate = True,
    #     save_raw_data = True, overwrite = False, debug = False)
    
    print('Finished!')











