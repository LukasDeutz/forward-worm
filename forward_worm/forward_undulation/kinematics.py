'''
Created on 14 Feb 2023

@author: lukas
'''

'''
Created on 9 Feb 2023

@author: lukas
'''

#Third-party imports
import h5py
import numpy as np
from scipy.integrate import trapezoid

# Local imports
from forward_worm.forward_undulation.undulations_dirs import log_dir, sim_dir, sweep_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5, sim_exp_wrapper

from simple_worm_experiments.forward_undulation.undulation import UndulationExperiment
from simple_worm_experiments.model_parameter import default_model_parameter
from simple_worm_experiments.experiment_post_processor import EPP

#from simple_worm_experiments.experiment_post_processor import EPP

from parameter_scan import ParameterGrid

#------------------------------------------------------------------------------ 
# Parameter

def undulation_rft_parameter():
    
    parameter = default_model_parameter()

    # Default kinematic parameter
    parameter['A'] = 5.0    
    parameter['lam'] = 1.0
    parameter['f'] = 2.0

    # Gradual muscle onset at tale and
    parameter['gmo'] = True              
    parameter['Ds_h'], parameter['Ds_t'] = 0.01, 0.01 
    parameter['s0_h'] = 3*parameter['Ds_h']
    parameter['s0_t'] = 1 - 3 * parameter['Ds_t']

    # Default fluid parameter
    parameter['external_force'] = ['rft']    
    parameter['rft'] = 'Whang'
    parameter['mu'] = 1e-3
    
    return parameter

#------------------------------------------------------------------------------ 
# Post processing 

def compute_swimming_speed(h5, PG):
        
    f = PG.base_parameter['f']
    T_undu = 1 / f
           
    x_arr, t = h5['FS']['x'], h5['t'][:]

    U_arr = np.zeros(x_arr.shape[0])
        
    for i, x in enumerate(x_arr):
        
        U_arr[i] = EPP.comp_mean_com_velocity(x, t, Delta_T = T_undu)    
    
    if 'U' in h5: h5['U'][:] = U_arr         
    else: h5.create_dataset('U', data = U_arr)
                        
    return

def compute_energy_per_undulation_period(
        h5, 
        PG, 
        N_undu_skip = 1):
    '''
    Compute energy consumption per undulation period
    
    :param h5 (h5py.File): hdf5 file
    :param PG (ParameterGrid): parameter grid object
    '''
       
    # Skipt N undulation periods to exlude initial 
    # transient from energy estimate 
    T_undu = 1.0 / PG.base_parameter['f']    
    powers, t = EPP.powers_from_h5(h5, t_start = N_undu_skip * T_undu)
    dt = t[1] - t[0]
        
    dot_V = powers['dot_V_k'] + powers['dot_V_sig']
    dot_D = powers['dot_D_k'] + powers['dot_D_sig']
    dot_W_F = powers['dot_W_F_F'] + powers['dot_W_F_T']
    dot_W_M = powers['dot_W_M_F'] + powers['dot_W_M_T']
    
    # Assume simulation time T is multiple of T_undu     
    T = PG.base_parameter['T']
    N_undu = np.round(T / T_undu) - N_undu_skip
    
    V = trapezoid(dot_V, dx=dt, axis = 1) / N_undu                                 
    D = trapezoid(dot_D, dx=dt, axis = 1) / N_undu                                 
    W_F = trapezoid(dot_W_F, dx=dt, axis = 1) / N_undu                                 
    W_M = trapezoid(dot_W_M, dx=dt, axis = 1) / N_undu                                 
    
    E_out = trapezoid(np.abs(- dot_V + dot_D + dot_W_F), dx=dt, axis = 1) / N_undu
    abs_W_M = trapezoid(np.abs(dot_W_M), dx=dt, axis = 1) / N_undu
                 
    grp = h5.create_group('energies')        
                
    grp.create_dataset('V', data = V)
    grp.create_dataset('D', data = D)
    grp.create_dataset('W_F', data = W_F)
    grp.create_dataset('W_M', data = W_M)
    grp.create_dataset('E_out', data = E_out)
    grp.create_dataset('abs_W_M', data = abs_W_M)
    
    # energy cost from "true" powers
    dot_V, dot_D, dot_W_F = EPP.comp_true_powers(powers)
    
    V = trapezoid(dot_V, dx=dt, axis = 1) / N_undu                                 
    D = trapezoid(dot_D, dx=dt, axis = 1) / N_undu                                 
    W_F = trapezoid(dot_W_F, dx=dt, axis = 1) / N_undu                                 
    E_out = V + D + W_F 

    grp = h5.create_group('energy-costs')
    grp.create_dataset('V', data = V)
    grp.create_dataset('D', data = D)
    grp.create_dataset('W_F', data = W_F)
    grp.create_dataset('E_out', data = E_out)
                
    return
    
#------------------------------------------------------------------------------ 
# Kinematic parameters

def sim_undulation_lam_c(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over wavelength lam for fixed dimensionless ratio c=A/q  
    '''    
    param = undulation_rft_parameter()
    param['T'] = 2.5
    param['N'] = 250
    param['dt'] = 1e-3
                
    # Parameter Grid    
    lam_min, lam_max, lam_step = 0.5, 2.0, 0.1
        
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 'N': None, 'step': lam_step, 'round': 2}                                                

    A_param = lam_param.copy()
    A_param['inverse'] = True
    A_param['round'] = 3
    
    c_arr = np.arange(0.5, 1.61, 0.1)

    for c in c_arr:
            
        A_param['scale'] = 2.0 * np.pi * c
                
        grid_param = {('lam', 'A'): (lam_param, A_param)}
    
        PG = ParameterGrid(param, grid_param)
        
        filename = (f'undulation_lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}'
            f'_c={c:.2f}_f={param["f"]}_mu_{param["mu"]}_N={param["N"]}_dt={param["dt"]}.h5')        
            
        h5_filepath = sweep_dir / filename 
        exper_spec = 'UE'
            
        h5 = sim_exp_wrapper(simulate, 
            save_raw_data,       
            ['x', 'Omega', 'sigma', 
            'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 
            'dot_W_F_lin', 'dot_W_F_rot', 
            'dot_W_M_lin', 'dot_W_M_rot'],
            ['Omega', 'sigma'],
            N_worker, 
            PG, 
            UndulationExperiment.sinusoidal_traveling_wave_control_sequence,
            h5_filepath,
            log_dir,
            sim_dir,
            exper_spec,
            overwrite,
            debug)
        
        compute_swimming_speed(h5, PG)
        compute_energy_per_undulation_period(h5, PG)
        
        h5.close()
                                                                                            
    return 

def sim_undulation_lam_c_eta_nu(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):

    lam_min, lam_max, lam_step = 0.5, 2.0, 0.1

    param = undulation_rft_parameter()
    param['T'] = 2.5
    param['N'] = 250
    param['dt'] = 1e-3
    param['N_report'] = 125
    param['dt_report'] = 1e-2
    param['A'] = None
                                
    # Parameter Grid    
    lam_min, lam_max, lam_step = 0.5, 2.0, 0.1        
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}                                                

    c_min, c_max, c_step = 0.5, 1.6, 0.1
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}                                                
        
    eta_min, eta_max, eta_step = -3, 0.0, 1.0
    
    eta_param = {'v_min': eta_min, 'v_max': eta_max + 0.1*eta_step, 
        'N': None, 'step': eta_step, 'round': 3, 'log': True, 'scale': param['E']}
    
    nu_param = eta_param.copy()
    nu_param['scale'] = param['G']

    grid_param = {'lam': lam_param, 'c': c_param, ('eta', 'nu'): (eta_param, nu_param)}

    PG = ParameterGrid(param, grid_param)

    filename = (
        f'undulation_lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={lam_min}_c_max={lam_max}_c_step={lam_step}_'
        f'eta_min={lam_min}_eta_max={lam_max}_eta_step={eta_step}_'                
        f'f={param["f"]}_mu_{param["mu"]}_N={param["N"]}_dt={param["dt"]}.h5')        
        
    h5_filepath = sweep_dir / filename 
    exper_spec = 'UE'
        
    h5 = sim_exp_wrapper(simulate, 
        save_raw_data,       
        ['x', 'Omega', 'sigma',
        'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 
        'dot_W_F_lin', 'dot_W_F_rot', 
        'dot_W_M_lin', 'dot_W_M_rot'],
        ['Omega', 'sigma'],
        N_worker, 
        PG, 
        UndulationExperiment.sinusoidal_traveling_wave_control_sequence,
        h5_filepath,
        log_dir,
        sim_dir,
        exper_spec,
        overwrite,
        debug)

    
    return
    
if __name__ == '__main__':
        
    N_worker = 16
        
    # sim_undulation_lam_c(N_worker, simulate = True,
    #         save_raw_data = True, overwrite = False, debug = False)    
    
    sim_undulation_lam_c_eta_nu(N_worker, simulate = True,
            save_raw_data = True, overwrite = False, debug = False)
    