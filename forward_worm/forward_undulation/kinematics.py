'''
Created on 14 Feb 2023

@author: lukas
'''
#Built-in imports
from sys import argv
from argparse import ArgumentParser
from itertools import repeat

#Third-party imports
import numpy as np
from scipy.integrate import trapezoid
import h5py

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

def save_swimming_speed_and_energy_to_h5(h5_filename, 
        h5_raw_data, PG, N_undu_skip = 1):
    
    h5_filepath = sweep_dir / h5_filename
    h5 = h5py.File(h5_filepath, 'w')
    h5.attrs['grid_filename'] = PG.filename + '.json' 

    U = compute_swimming_speed(h5_raw_data, PG, N_undu_skip)
    energies = compute_energies(h5_raw_data, PG, N_undu_skip)
    
    h5.create_dataset('U', data=U)
    
    grp = h5.create_group('energies')    
    for k, e in energies.items(): grp.create_dataset(k, data = e)
            
    h5.close()
    print(f'Saved swimming speed and energies to {h5_filepath}')
    
    return
    
def compute_swimming_speed(h5, PG, N_undu_skip = 1):

    if PG.has_key('T') and PG.has_key('f'):
        U = compute_swimming_speed_for_varying_f_and_T(  
            h5, PG, N_undu_skip)
    else: 
        U = compute_swimming_speed_for_fixed_f_and_T(  
            h5, PG, N_undu_skip)

    return U 

def compute_swimming_speed_for_fixed_f_and_T(
        h5, 
        PG,
        N_undu_skip):
                        
    f = PG.base_parameter['f']
    T_undu = 1.0 / f
            
    x_arr = h5['FS']['x'][:]
    t = h5['t'][:]    
    U_arr = np.zeros(x_arr.shape[0])
            
    for i, x in enumerate(x_arr):
                                                
        U_arr[i] = EPP.comp_mean_com_velocity(x, t, Delta_T = N_undu_skip * T_undu)    

    return U_arr.reshape(PG.shape)

def compute_swimming_speed_for_varying_f_and_T(
        h5, 
        PG,
        N_undu_skip = 1):
            
    f_arr = PG.v_from_key('f')
    T_arr = PG.v_from_key('T')
    U_arr_list = []
                                           
    for T, f in zip(T_arr, f_arr):
        
        x_arr = h5['FS']['x'][f'{T}'][:]
        t = h5['t'][f'{T}'][:]
        T_undu = 1.0 / f                                                
        U_arr = np.zeros(x_arr.shape[0])
                                                       
        for i, x in enumerate(x_arr):
                                 
            U_arr[i] = EPP.comp_mean_com_velocity(x, 
                t, Delta_T = N_undu_skip * T_undu)    
        
        U_arr_list.append(U_arr.reshape(PG.shape[1:]))
                                    
    return np.array(U_arr_list)

def compute_energies(
        h5, 
        PG, 
        N_undu_skip = 1):

    if PG.has_key('T') and PG.has_key('f'):
        energies = compute_energies_for_varying_f_and_T(
            h5, PG, N_undu_skip = 1)
    else: 
        energies = compute_energies_for_fixed_f_and_T(
            h5, PG, N_undu_skip = 1)

    return energies
    
def compute_energies_for_fixed_f_and_T(
        h5, 
        PG, 
        N_undu_skip = 1):
    '''
    Compute energy consumption per undulation period
    
    :param h5 (h5py.File): hdf5 file
    :param PG (ParameterGrid): parameter grid object
    '''              
    
    T = PG.base_parameter['T']
    f = PG.base_parameter['f']
    T_undu = 1.0 / f                
    t_start = N_undu_skip * T_undu 

    # Assume that simulation time T is multiple of T_undu     
    N_undu = np.round(T / T_undu) - N_undu_skip
            
    # Get powers        
    power_grp = h5.create_group('powers')
    
    powers, t = EPP.powers_from_h5(h5, t_start=t_start)

    dot_D_I = powers['dot_D_I_k'] + powers['dot_D_I_sig']
    dot_D_F = powers['dot_D_F_F'] + powers['dot_D_F_T']
    dot_W =  powers['dot_W_M_F'] + powers['dot_W_M_T']

    dt = t[1]-t[0]
                                                                                                  
    # Compute energies        
    D_I = trapezoid(dot_D_I, dx=dt, axis = 1) / N_undu
    D_F = trapezoid(dot_D_F, dx=dt, axis = 1) / N_undu
    W = trapezoid(dot_W, dx=dt, axis = 1) / N_undu
           
    energies = {'D_I': D_I, 'D_F': D_F, 'W': W}
                                                                                                              
    return energies

def compute_energies_for_varying_f_and_T(
        h5,
        PG,
        N_undu_skip = 1):

    T_arr = PG.v_from_key('T')
    f_arr = PG.v_from_key('f')
    
    T_undu_arr = 1.0 / f_arr
    t_start_arr = N_undu_skip * T_undu_arr 

    # Assume that simulation time T is multiple of T_undu     
    N_undu_arr = np.round(T_arr / T_undu_arr) - N_undu_skip
        
    D_I_list, D_F_list, W_list = [], [], []
                
    for i, (T, t_start, N_undu) in enumerate(zip(T_arr, t_start_arr, N_undu_arr)): 
        
        powers, t = EPP.powers_from_h5(h5, t_start=t_start, T=T)
                
        dot_V = powers['dot_V_k'] + powers['dot_V_sig']        
        dot_D_I = powers['dot_D_I_k'] + powers['dot_D_I_sig']        
        dot_D_F = powers['dot_D_F_F'] + powers['dot_D_F_T']        
        dot_W = powers['dot_W_M_F'] + powers['dot_W_M_T']        
                                                                                                    
        # Energies
        dt = t[1]-t[0]
                
        D_I = trapezoid(dot_D_I, dx=dt, axis = 1) / N_undu
        D_F = trapezoid(dot_D_F, dx=dt, axis = 1) / N_undu
        W = trapezoid(dot_W, dx=dt, axis = 1) / N_undu
        
        D_I_list.append(D_I.reshape(PG.shape[1:]))
        D_F_list.append(D_F.reshape(PG.shape[1:]))
        W_list.append(W.reshape(PG.shape[1:]))
        
    D_I = np.array(D_I_list)
    D_F = np.array(D_F_list)
    W = np.array(W_list)
            
    energies = {'D_I': D_I, 'D_F': D_F, 'W': W}
                                                    
    return energies
        
def sim_mu_E(argv):
    '''
    Parameter sweep over wavelength lam amplitude wavenumber ratio c=A/q  
    '''    
    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    
    sweep_parser.add_argument('--mu', 
        type=float, nargs = 3, default=[-3, 1, 5])    
    sweep_parser.add_argument('--E', 
        type=float, nargs = 3, default=[1, 7, 30])    
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = model_parameter_parser()    
    model_param = vars(model_parser.parse_known_args(argv)[0])

    # print parameter which have been set
    print({k: v for k, v in 
        model_param.items() if v != model_parser.get_default(k)})

    E_min, E_max = sweep_param.E[0], sweep_param.E[1]   
    N_E = sweep_param.E[2]
    mu_min, mu_max = sweep_param.mu[0], sweep_param.mu[1]
    N_mu = sweep_param.mu[2]

    mu_param = {'v_min': mu_min, 'v_max': mu_max, 
        'N': N_mu, 'step': None, 'round': 3, 'log': True}    
                        
    E_param = {'v_min': E_min, 'v_max': E_max, 
        'N': N_E, 'step': None, 'round': 0, 'log': True}    

    # Poisson ratio
    p = 0.5    
    G_param = E_param.copy()
    G_param['scale'] = 1.0 / (2.0 * (1 + p) )
    
    eta_param = E_param.copy()
    eta_param['scale'] = 1e-2
    eta_param['round'] = 2

    nu_param = E_param.copy()
    nu_param['scale'] = 1.0 / (2.0 * (1 + p) ) * 1e-2
    nu_param['round'] = 2
    
    grid_param = {'mu': mu_param,
        ('E', 'G', 'eta', 'nu'): (E_param, G_param, eta_param, nu_param)}

    PG = ParameterGrid(model_param, grid_param)

    # Run sweep
    filename = (
        f'raw_data_mu_'
        f'mu_min={mu_min}_mu_max={mu_max}_N_mu={N_mu}_'
        f'E_min={E_min}_E_max={E_max}_N_E={N_E}_'
        f'A_{model_param["A"]}_lam_{model_param["lam"]}_f={model_param["f"]}_'
        f'N={model_param["N"]}_dt={model_param["dt"]}.h5')        

    h5_filepath = sweep_dir / filename 
    exper_spec = 'UE'
        
    h5_raw_data = sim_exp_wrapper(sweep_param.simulate, 
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

    h5_filename = (f'speed_and_energies_'
        f'mu_min={mu_min}_mu_max={mu_max}_N_mu={N_mu}_'
        f'E_min={E_min}_E_max={E_max}_N_E={N_E}_'
        f'A_{model_param["A"]}_lam_{model_param["lam"]}_f={model_param["f"]}_'
        f'N={model_param["N"]}_dt={model_param["dt"]}.h5')        

    save_swimming_speed_and_energy_to_h5(h5_filename, 
        h5_raw_data, PG, N_undu_skip = 1)
    
    h5_raw_data.close()
            
    return
    
def sim_lam_c(argv):
    '''
    Parameter sweep over wavelength lam amplitude wavenumber ratio c=A/q  
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
    model_param = vars(model_parser.parse_known_args(argv)[0])
    
    # print parameter which have been set
    print({k: v for k, v in 
        model_param.items() if v != model_parser.get_default(k)})
                        
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
        f'raw_data_lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
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

def sim_f_lam_c(argv):
    '''
    Parameter sweep over wavelength lam amplitude wavenumber ratio c=A/q and undulation frequency  
    '''    
    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    
    sweep_parser.add_argument('--lam', 
        type=float, nargs = 3, default=[0.5, 2.0, 0.1])    
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default=[0.5, 1.6, 0.1])     
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    sweep_parser.add_argument('--f', 
        type=float, nargs=3, default=[0.5, 2.0, 0.5])     
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = model_parameter_parser()    
    model_param = vars(model_parser.parse_known_args(argv)[0])
                                
    # Creare parameter Grid            
    f_min, f_max = sweep_param.f[0], sweep_param.f[1]
    f_step = sweep_param.f[2] 
    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]   
    lam_step = sweep_param.lam[2]        
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2] 
            
    f_param = {'v_min': f_min, 'v_max': f_max + 0.1*f_step, 
        'N': None, 'step': f_step, 'round': 2}                                                
    T_param = f_param.copy()
    T_param['inverse'] = True
    T_param['scale'] = 5.0        
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}                                                
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}                                                
    
    grid_param = {('f', 'T'): (f_param, T_param), 'lam': lam_param, 'c': c_param}

    PG = ParameterGrid(model_param, grid_param)

    filename = (
        f'raw_data_'
        f'f_min={f_min}_f_max={f_max}_f_step={f_step}_'                
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={lam_min}_c_max={lam_max}_c_step={lam_step}_'
        f'mu_{model_param["mu"]}_N={model_param["N"]}_dt={model_param["dt"]}.h5')        
        
    h5_filepath = sweep_dir / filename 
    exper_spec = 'UE'
        
    h5_raw_data = sim_exp_wrapper(sweep_param.simulate, 
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

    h5_filename = (f'speed_and_energies_'
        f'f_min={f_min}_f_max={f_max}_f_step={f_step}_'                
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={lam_min}_c_max={lam_max}_c_step={lam_step}_'
        f'mu_{model_param["mu"]}_N={model_param["N"]}_dt={model_param["dt"]}.h5')        

    save_swimming_speed_and_energy_to_h5(h5_filename, 
            h5_raw_data, PG, N_undu_skip = 1)

    h5_raw_data.close()
    
    return

def sim_eta_nu_lam_c(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):

    '''
    Parameter sweep over wavelength lam amplitude wavenumber ratio c=A/q and undulation frequency  
    '''    
    
    # parse sweep parameter
    sweep_parser = default_sweep_parameter()    
    sweep_parser.add_argument('--eta', 
        type=float, nargs=3, default=[-3, 0.0, 1.0])     
    sweep_param = sweep_parser.parse_known_args(argv)[0]    
    sweep_parser.add_argument('--lam', 
        type=float, nargs = 3, default=[0.5, 2.0, 0.1])    
    sweep_parser.add_argument('--c', 
        type=float, nargs=3, default=[0.5, 1.6, 0.1])     
    sweep_param = sweep_parser.parse_known_args(argv)[0]    

    # parse model parameter and convert to dict
    model_parser = model_parameter_parser()    
    model_param = vars(model_parser.parse_known_args(argv)[0])
                                
    # Create parameter Grid            
    eta_min, eta_max = sweep_param.eta[0], sweep_param.eta[1]
    eta_step = sweep_param.eta[2] 
    lam_min, lam_max = sweep_param.lam[0], sweep_param.lam[1]   
    lam_step = sweep_param.lam[2]        
    c_min, c_max = sweep_param.c[0], sweep_param.c[1]
    c_step = sweep_param.c[2] 
                                
    eta_param = {'v_min': eta_min, 'v_max': eta_max + 0.1*eta_step, 
        'N': None, 'step': eta_step, 'round': 3, 'log': True, 'scale': model_param['E']}    
    nu_param = eta_param.copy()
    nu_param['scale'] = model_param['G']
    lam_param = {'v_min': lam_min, 'v_max': lam_max + 0.1*lam_step, 
        'N': None, 'step': lam_step, 'round': 2}                                                
    c_param = {'v_min': c_min, 'v_max': c_max + 0.1*c_step, 
        'N': None, 'step': c_step, 'round': 2}                                                
    
    grid_param = {('eta', 'nu'): (eta_param, nu_param), 'lam': lam_param, 'c': c_param}

    PG = ParameterGrid(model_param, grid_param)

    filename = (
        f'undulation_'
        f'eta_min={eta_min}_eta_max={eta_max}_eta_step={eta_step}_'                
        f'lam_min={lam_min}_lam_max={lam_max}_lam_step={lam_step}_'
        f'c_min={lam_min}_c_max={lam_max}_c_step={lam_step}_'
        f'f={model_param["f"]}_mu_{model_param["mu"]}_N={model_param["N"]}_dt={model_param["dt"]}.h5')        
        
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
    
if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument('-sim',  
        type = str, 
        choices = ['sim_mu_E', 'sim_lam_c', 'sim_f_lam_c', 'sim_eta_nu_lam_c'], 
        help='Simulation function to call')
    # Run function passed via command line
    args = parser.parse_known_args(argv)[0]    
    globals()[args.sim](argv)
        
    