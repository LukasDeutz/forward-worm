'''
Created on 14 Feb 2023

@author: lukas
'''
from sys import argv
from argparse import ArgumentParser

#Third-party imports
import numpy as np
from scipy.integrate import trapezoid

# Local imports
from forward_worm.forward_undulation.undulations_dirs import log_dir, sim_dir, sweep_dir
from forward_worm.util import sim_exp_wrapper
from forward_worm.util import default_sweep_parameter

from simple_worm_experiments.model_parameter import default_model_parameter
from simple_worm_experiments.forward_undulation.undulation import UndulationExperiment
from simple_worm_experiments.experiment_post_processor import EPP
from parameter_scan import ParameterGrid

#------------------------------------------------------------------------------ 
# Parameter
    
def model_parameter_parser():
    
    model_param = default_model_parameter(as_dict=False)
    
    model_param.add_argument('--T', type=float, default=2.5,
        help='Simulation time')
    
    # Muscle timescale
    model_param.add_argument('--fmts', type=bool, default=True,
        help='If true, muscles switch on on a finite time scale')
    model_param.add_argument('--tau_on', type=float, default = 0.1,
        help='Muscle time scale')
    model_param.add_argument('--t0_on', type=float, default = 5*0.1,
        help='Sigmoid midpoint')
        
    # Kinematic parameter    
    model_param.add_argument('--f', type=float, default=2.0,
        help='Undulation frequency')
    model_param.add_argument('--A', type=float, default=None,
        help='Undulation amplitude')
    model_param.add_argument('--lam', type=float, default=1.0,
        help='Undulation wavelength')
    
    return model_param

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
    
def sim_lam_c(argv):
    '''
    Parameter sweep over wavelength lam for fixed dimensionless ratio c=A/q  
    '''    
    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    
    sweep_parser.add_argument('--lam', 
        type=float, nargs = 3, default=[0.5, 2.0, 0.1])    
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default=[0.5, 1.6, 0.1])     
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = model_parameter_parser()    
    model_parser.print_help()
    model_param = vars(model_parser.parse_known_args(argv)[0])
                    
    # Creare parameter Grid            
    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]   
    lam_step = sweep_param.lam[2]        
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2] 
                
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}                                                
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}                                                
    
    grid_param = {'lam': lam_param, 'c': c_param}

    PG = ParameterGrid(model_param, grid_param)

    # Run sweep
    filename = (
        f'undulation_lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={c_min}_c_max={c_max}_c_step={c_step}_'
        f'f={model_param["f"]}_mu_{model_param["mu"]}_'
        f'N={model_param["N"]}_dt={model_param["dt"]}.h5')        

    h5_filepath = sweep_dir / filename 
    exper_spec = 'UE'
        
    h5 = sim_exp_wrapper(sweep_param.simulate, 
        sweep_param.save_raw_data,       
        ['x', 'Omega', 'sigma',
        'V_dot_k', 'V_dot_sig', 'D_k', 'D_sig', 
        'dot_W_F_lin', 'dot_W_F_rot', 
        'dot_W_M_lin', 'dot_W_M_rot'],
        ['Omega', 'sigma'],
        sweep_param.worker, 
        PG, 
        UndulationExperiment.sinusoidal_traveling_wave_control_sequence,
        h5_filepath,
        log_dir,
        sim_dir,
        exper_spec,
        sweep_param.overwrite,
        sweep_param.debug)
    
    compute_swimming_speed(h5, PG)
    compute_energy_per_undulation_period(h5, PG)
        
    h5.close()
                                                                                            
    return 

def sim_lam_c_eta_nu(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):

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

    h5.close()
    
    return
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-sim',  
        type = str, 
        choices = ['sim_lam_c', 'sim_lam_c_eta_nu'], 
        help='Simulation function to call')
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()[args.sim](argv)
        
    