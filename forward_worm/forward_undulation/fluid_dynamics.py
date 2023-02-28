'''
Created on 9 Feb 2023

@author: lukas
'''

#Built-in imports
from os.path import join

#Third-party imports
import h5py

# Local imports
from forward_worm.forward_undulation.undulations_dirs import log_dir, output_dir, sweep_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5, sim_exp_wrapper

from simple_worm_experiments.forward_undulation.undulation import UndulationExperiment
#from simple_worm_experiments.experiment_post_processor import EPP

from parameter_scan import ParameterGrid

#------------------------------------------------------------------------------ 
# Parameter

def base_model_parameter():
    
    parameter = {}
    
    # Simulation parameter          
    parameter['N']  = 100
    parameter['dt'] = 0.01
    parameter['T'] = 2.5
    parameter['N_report'] = 100
    parameter['dt_report'] = 0.01

    # Solver parameter
    parameter['pi'] = False
    parameter['pi_alpha0'] = 0.9            
    parameter['pi_maxiter'] = 1000             
    parameter['fdo'] = {1: 2, 2:2}
      
    # Model parameter    
    parameter['use_inertia'] = False
                    
    # Geometric parameter    
    parameter['L0'] = 1130 * 1e-6
    parameter['r_max'] = 32 * 1e-6
    parameter['rc'] = 'spheroid'
    parameter['eps_phi'] = 1e-3
            
    # Material parameter    
    parameter['E'] = 1e5  
    parameter['G'] = parameter['E'] / (2 * (1 + 0.5))
    parameter['eta'] = 1e-2 * parameter['E']
    parameter['nu'] = 1e-2 * parameter['G']

    # Muscle parameter            
    parameter['fmts'] = True
    parameter['tau_on'] = 0.05
    parameter['Dt_on'] = 3*parameter['tau_on']
        
    parameter['gmo'] = True    
    parameter['Ds_h'] = 0.01
    parameter['s0_h'] = 3*parameter['Ds_h']    
    parameter['Ds_t'] = 0.01
    parameter['s0_t'] = 1 - 3*parameter['Ds_t']
        
    return parameter

def undulation_rft_parameter():
    
    parameter = base_model_parameter()

    # Default kinematic parameter
    parameter['A'] = 5.0    
    parameter['lam'] = 1.0
    parameter['f'] = 2.0

    # Fluid parameter
    parameter['external_force'] = ['rft']    
    parameter['rft'] = 'Whang'
    parameter['mu'] = 1e-3
    
    return parameter

#------------------------------------------------------------------------------ 
# Simulations

def sim_undulation_rft_A(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over curvature amplitude
    '''    
    param = undulation_rft_parameter()
    param['T'] = 2.5
        
    # Parameter Grid    
    A_min, A_max, A_step = 1.0, 8.0, 1.0
        
    A_param = {'v_min': A_min, 'v_max': A_max + 0.1*A_step, 'N': None, 'step': A_step, 'round': 1}                                                
    grid_param = {'A': A_param}
    
    PG = ParameterGrid(param, grid_param)
        
    filename = (f'undulation_rft_A_min={A_min}_A_max={A_max}_A_step={A_step}'
        f'_f={param["f"]}_mu={param["mu"]}_N={param["N"]}_dt={param["dt"]}.h5')        
    
    h5_filepath = sweep_dir / filename 
    exper_spec = 'UE'
        
    h5 = sim_exp_wrapper(simulate, 
        save_raw_data,       
        ['x', 'x_t', 'theta', 'w', 'e1', 'e2', 'e3', 'f', 'l'],
        [],
        N_worker, 
        PG, 
        UndulationExperiment.sinusoidal_traveling_wave_control_sequence,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)
        
    A_arr = PG.v_from_key('A')        
    h5.create_dataset('A', data = A_arr)
                                                                                    
    return 

def sim_undulation_lam(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over curvature amplitude
    '''    
    parameter = undulation_rft_parameter()        
    
    # Parameter Grid    
    lam_min, lam_max, lam_step = 0.5, 2.0, 0.2
        
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 'N': None, 'step': lam_step, 'round': 2}                                                
    grid_param = {'lam': lam_param}
    
    PG = ParameterGrid(parameter, grid_param)
        
    filename = (f'continuous_roll_lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}'
        f'_f={parameter["f"]}_N={parameter["N"]}_dt={parameter["dt"]}.h5')        
   
    h5_filepath = sweep_dir / filename        
    exper_spec = 'UE'

    PG.save(log_dir)
        
    if simulate:        
        PGL = simulate_experiments(N_worker, PG, 
            UndulationExperiment.sinusoidal_traveling_wave_control_sequence, 
            log_dir, output_dir, exper_spec = exper_spec, overwrite = overwrite, 
            debug = debug)

        PGL.close()
                
    if save_raw_data: # Save raw simulation results to h5                                     
        h5 = save_experiment_to_h5(PG, 
            str(h5_filepath), output_dir, log_dir, 
            FS_keys = ['x', 'x_t', 'theta', 'e1', 'e2', 'e3', 'w'], 
            CS_keys = [])    
    else:        
        assert not simulate, 'If simulate is True, save_raw_data must be True'
        h5 = h5py.File(h5_filepath, 'r+')    
        
    A_arr = PG.v_from_key('A')        
    h5.create_dataset('A', data = A_arr)
                                                                                    
    return 



if __name__ == '__main__':
    
    N_worker = 8
    
    sim_undulation_rft_A(N_worker, simulate = False, save_raw_data = True,
        overwrite = False, debug = True)    
    
    
    
    
    

