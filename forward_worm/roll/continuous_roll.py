'''
Created on 11 Jan 2023

@author: lukas
'''

#Built-in imports
from os.path import join

#Third-party imports
import numpy as np
import h5py

# Local imports
import forward_worm.roll.roll_parameter as roll_parameter
from forward_worm.roll.roll_parameter import data_path, log_dir, output_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5
from simple_worm_experiments.roll.roll import RollManeuverExperiment
from simple_worm_experiments.grid_loader import GridLoader, GridPoolLoader
from simple_worm_experiments.experiment_post_processor import EPP
from parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid
from mp_progress_logger import FWProgressLogger 

    
def sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        CS_func,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug):
    '''
    Runs parameter sweep over given parameter grid
    '''
        
    PG.save(log_dir)
        
    if simulate:        
        PGL = simulate_experiments(N_worker, 
            PG, 
            CS_func,
            log_dir,
            output_dir,
            exper_spec = exper_spec,
            overwrite = overwrite,
            debug = debug)

        PGL.close()
                
    if save_raw_data: # Save raw simulation results to h5                                     
        h5 = save_experiment_to_h5(PG, 
            h5_filepath,
            output_dir,
            log_dir, 
            FS_keys = ['x', 'theta', 'Omega', 'e1', 'e2', 'e3', 'w'], 
            CS_keys = [])
    
    else:        
        assert not simulate, 'If simulate is True, save_raw_data must be True'
        h5 = h5py.File(h5_filepath, 'r+')    
    
    return h5

def get_s_mask(parameter):
    '''
    Returns mask for the worm's body position which
    identifies the tale
    '''
    s_arr = np.linspace(0, 1, parameter['N'])
    s_mask = s_arr >= parameter['s0_t']

    return s_mask

def linear_map_v2T(v1, v2, T1, T2, v_step,
        v1_map = None, 
        v2_map = None):
    '''
    Returns simulation times assuming a linear relationship between 
    given model parameter and desired simulation time
    '''
    if v1_map is None: v1_map = v1
    if v2_map is None: v2_map = v2
                            
    DT, Dv = T2 - T1, v2_map - v1_map            
    a = DT / Dv
    b = T1 - a * v1_map    
    
    v_arr = np.arange(v1, v2 + 0.1*v_step, v_step)
        
    T_arr = a * v_arr + b
    
    return T_arr.tolist()

def compute_roll_frequency_from_euler_angle(h5, PG):
    '''
    Computes the average roll frequency of the worm's body
    using the time evolution of body averaged roll angle     
    '''

    FS_grp = h5['FS']    
    theta = FS_grp['theta']

    # Fixed simulation time, FrameSequence keys 
    # for entire sweep are saved in one dataset        
    if isinstance(FS_grp['e2'], h5py.Dataset):
        ALPHA = FS_grp['theta'][:, :, 0, :]            
        T = len(ALPHA)*[h5['t'][:]] 
    # If simulation time changes over the grid,
    # then FrameSequence keys for every grid point
    # are saved in seperate datasets 
    else:            
        T = [h5['t'][h][:] for h in PG.hash_arr]
        ALPHA = [FS_grp['theta'][h][:, 0, :] for h in PG.hash_arr] 
    
    s_mask = get_s_mask(PG.base_parameter)
    
    f_avg_arr = np.zeros(len(ALPHA))
    f_std_arr = np.zeros(len(ALPHA))
                        
    for i, (alpha, t) in enumerate(zip(ALPHA, T)):
    
        f_avg, f_std = EPP.comp_roll_frequency_from_euler_angle(
            alpha, 
            t, 
            Dt = 0.25)
        
        f_avg_arr[i] = f_avg
        f_std_arr[i] = f_std
    
    if 'f_avg_euler' in h5: h5['f_avg_euler'][:] = f_avg_arr    
    else: h5.create_dataset('f_avg_euler', data = f_avg_arr)    
    if 'f_std_euler' in h5: h5['f_std_euler'][:] = f_std_arr    
    else: h5.create_dataset('f_std_euler', data = f_std_arr)
    
    if 'alpha' in h5: 
        grp = h5['alpha']
        for alpha, dataset in zip(ALPHA, grp.values()):
            dataset[:] = alpha
        
    else: 
        grp = h5.create_group('alpha')
        for alpha, h in zip(ALPHA, PG.hash_arr):                        
            grp.create_dataset(h, data = alpha)
        
    return
    
def compute_roll_frequency(h5, PG):
    '''
    Computes the average roll frequency of the worm's body
    using spherical representation of the body frame 
    vectors with respect to a fixed reference frame. 
    
    Outdated: Euler angle method is preferable method
    '''
            
    FS_grp = h5['FS']

    # Fixed simulation time, FrameSequence keys 
    # for entire sweep are saved in one dataset        
    if isinstance(FS_grp['e2'], h5py.Dataset):
        d1_ref = FS_grp['e1'][0, 0, :, 0]
        d2_ref = FS_grp['e2'][0, 0, :, 0]
        d3_ref = FS_grp['e3'][0, 0, :, 0]                    
        
        D2 = FS_grp['e2'][:]
        T = len(D2)*[h5['t'][:]]
    
    # If simulation time changes over the grid,
    # then FrameSequence keys for every grid point
    # are saved in seperate datasets 
    else:
        d1_ref = list(FS_grp['e1'].values())[0][0, :, 0] 
        d2_ref = list(FS_grp['e2'].values())[0][0, :, 0] 
        d3_ref = list(FS_grp['e3'].values())[0][0, :, 0]
        
        T = [h5['t'][h][:] for h in PG.hash_arr]
        D2 = [FS_grp['e2'][h][:] for h in PG.hash_arr] 
    
    d123_ref = np.vstack((d1_ref, d2_ref, d3_ref))
    s_mask = get_s_mask(PG.base_parameter)
            
    f_avg_arr = np.zeros(len(D2))
    f_std_arr = np.zeros(len(D2))
        
    phi_list = []
                
    for i, (d2, t) in enumerate(zip(D2, T)):
    
        f_avg, f_std, phi = EPP.comp_roll_frequency_from_spherical_angle(
                d2, t, d123_ref, None, Dt = None)
        
        
        f_avg_arr[i] = f_avg
        f_std_arr[i] = f_std
        phi_list.append(phi) 

    if 'f_avg' in h5: h5['f_avg'][:] = f_avg_arr    
    else: h5.create_dataset('f_avg', data = f_avg_arr)    
    if 'f_std' in h5: h5['f_std'][:] = f_std_arr    
    else: h5.create_dataset('f_std', data = f_std_arr)
    
    if 'phi' in h5: 
        grp = h5['phi']
        for phi, dataset in zip(phi_list, grp.values()):
            dataset[:] = phi
        
    else: 
        grp = h5.create_group('phi')
        for phi, h in zip(phi_list, PG.hash_arr):                        
            grp.create_dataset(h, data = phi)
        
    return

#------------------------------------------------------------------------------ 
# Kinematic parameters
    
def sim_continuous_roll_A(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over curvature amplitude
    '''
    
    parameter = roll_parameter.get_continuous_roll_parameter()
    
    N = 200
    dt = 0.001
    parameter['dt'] = dt
    parameter['N'] = N
        
    # Parameter Grid    
    A_min, A_max, A_step = 1.0, 7.0, 1.0
    T_min, T_max = 5.0, 40.0
    N_A = int((A_max - A_min) / A_step) + 1
        
    A_dv_param = {'v_min': A_min, 'v_max': A_max + 0.1*A_step, 'N': None, 'step': A_step, 'round': 2}        
    A_lr_param = A_dv_param.copy()    
                
    T_param = {'v_min': T_min, 'v_max': T_max, 'N': N_A, 'step': None, 'round': 1, 'flip_lr': True}
                            
    grid_param = {('A_dv', 'A_lr', 'T') : (A_dv_param, A_lr_param, T_param)}
    
    PG = ParameterGrid(parameter, grid_param)
        
    s0_h = parameter['s0_h']
    s0_t = parameter['s0_t']              
    filename = f'continuous_roll_A_min={A_min}_A_max={A_max}_A_step={A_step}_N_{N}_dt_{dt}_s0_h_{s0_h}_s0_t_{s0_t}.h5'        
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'RRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)

    # Analyse raw data and save results
    compute_roll_frequency(h5, PG)
    compute_roll_frequency_from_euler_angle(h5, PG)

                                                                                    
    return

def sim_continuous_roll_f(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over muscle frequency
    '''
    # Default parameter
    parameter = roll_parameter.get_continuous_roll_parameter()
        
    # Parameter grid to sweep over
    f_min, f_max, f_step = 1.0, 8.0, 1.0
    T_min, T_max = 5.0, 40.0
    N_f = int((f_max - f_min) / f_step) + 1
    f_dv_param = {'v_min': f_min, 'v_max': f_max + 0.1*f_step, 'N': None, 'step': f_step, 'round': 2}        
    f_lr_param = f_dv_param.copy()    
    T_param = {'v_min': T_min, 'v_max': T_max, 'N': N_f, 'step': None, 'round': 1, 'flip_lr': True}
                            
    grid_param = {('f_dv', 'f_lr', 'T') : (f_dv_param, f_lr_param, T_param)}
    
    PG = ParameterGrid(parameter, grid_param)
      
    filename = (f'continuous_roll_f_min={f_min}_f_max={f_max}_f_step={f_step}'
    f'_A={parameter["A_dv"]}_s0_h={parameter["s0_h"]}_s0_t_{parameter["s0_t"]}'
    f'_N={parameter["N"]}_dt={parameter["dt"]}.h5')
        
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)

    # Analyse raw data and save results
    compute_roll_frequency(h5, PG)
    compute_roll_frequency_from_euler_angle(h5, PG)
                                                                                    
    return   

def sim_continuous_roll_s0_t(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Parameter sweep over "neck" position
    '''
    # default parameter
    parameter = roll_parameter.get_continuous_roll_parameter()
    parameter['dt'] = 0.01
    parameter['N'] = 100
    parameter['T'] = 10.0
    parameter['A_dv'] = 5.0
    parameter['A_lr'] = 5.0
                          
    # Parameter grid to sweep over
    s0_t_min, s0_t_max, s0_t_step = 0.1, 0.3, 0.01 
    T_min, T_max = 5.0, 40.0
    N_s0_t = int( (s0_t_max - s0_t_min) / s0_t_step + 0.1*s0_t_step) + 1
    T_param = {'v_min': T_min, 'v_max': T_max, 'N': N_s0_t, 'step': None, 'round': 1, 'flip_lr': True}
    
    s0_t_param = {'v_min': s0_t_min, 'v_max': s0_t_max + 0.1*s0_t_step, 'N': None, 'step': s0_t_step, 'round': 2}                                    
    
    grid_param = {('s0_t', 'T'): (s0_t_param, T_param)}    
    PG = ParameterGrid(parameter, grid_param)
      
    filename = (f'continuous_roll_s0_t_min={s0_t_min}_s0_t_max={s0_t_max}_s0_t_step={s0_t_step}_'
    f'A={parameter["A_dv"]}_f={parameter["f_dv"]}_N={parameter["N"]}_dt={parameter["dt"]}.h5')
        
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)

    # Analyse raw data and save results
    compute_roll_frequency(h5, PG)
    compute_roll_frequency_from_euler_angle(h5, PG)
                                                                                    
    return   
    
def sim_continuous_roll_A_f(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):

    parameter = roll_parameter.get_continuous_roll_parameter()

    N = 100
    dt = 0.01    
    parameter['N'] = N    
    parameter['dt'] = 0.01

    # Parameter Grid    
    A_min, A_max, A_step = 1.0, 7.0, 1.0
    T_min, T_max = 5.0, 40.0
    N_A = int((A_max - A_min) / A_step) + 1         
    A_dv_param = {'v_min': A_min, 'v_max': A_max + 0.1*A_step, 'N': None, 'step': A_step, 'round': 2}        
    A_lr_param = A_dv_param.copy()    
    T_param = {'v_min': T_min, 'v_max': T_max, 'N': N_A, 'step': None, 'round': 1, 'flip_lr': True}

    f_min, f_max, f_step = 2.0, 8.0, 2        
    f_dv_param = {'v_min': f_min, 'v_max': f_max + 0.1*f_step, 'N': None, 'step': f_step, 'round': 0}        
    f_lr_param = f_dv_param.copy()    
                                
    grid_param = {('A_dv', 'A_lr', 'T') : (A_dv_param, A_lr_param, T_param), 
                  ('f_dv', 'f_lr') : (f_dv_param, f_lr_param)}
    
    PG = ParameterGrid(parameter, grid_param)
    
    s0_h = parameter['s0_h']
    s0_t = parameter['s0_t']            
    filename = (f'continuous_roll_A_f_A_min={A_min}_A_max={A_max}_A_step={A_step}'
    f'_f_min={f_min}_f_max={f_max}_f_step={f_step}_N_{N}_dt_{dt}_s0_h_{s0_h}_s0_t_{s0_t}.h5')               
    
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'RRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)
    
    # Analyse raw data and save results
        
    compute_roll_frequency(h5, PG)    
    compute_roll_frequency_from_euler_angle(h5, PG)

    return

#------------------------------------------------------------------------------ 
# Fluid viscosity

def sim_continuous_roll_mu(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Simulate continuous roll experiments for different fluid viscosity mu    
    '''

    parameter = roll_parameter.get_continuous_roll_parameter()

    N, dt = 100, 0.01
    parameter['N'], parameter['dt'] = N, dt    
    mu_min, mu_max = -3.0, 0.0
    N_mu = 10 
    T_min, T_max = 5.0, 20

    # Parameter Grid    
    mu_param = {'v_min': mu_min, 'v_max': mu_max, 'N': N_mu, 
        'step': None, 'round': 3, 'log': True}        
                                
    T_param = {'v_min': T_min, 'v_max': T_max, 'N': N_mu, 
        'step': None, 'round': 1, 'flip_lr': False}
        
    grid_param = {('mu','T') : (mu_param, T_param)}
    
    PG = ParameterGrid(parameter, grid_param)
                        
    filename = (f'continuous_roll_mu_min={mu_min}_mu_max={mu_max}_N_mu={N_mu}_'
    f'A_={parameter["A_dv"]}_f={parameter["f_dv"]}_s0_t={parameter["s0_t"]}_'
    f's0_h={parameter["s0_h"]}_s0_t={parameter["s0_t"]}_N_{N}_dt_{dt}.h5')               
    
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)
    
    # Analyse raw data and save results        
    compute_roll_frequency(h5, PG)    
    compute_roll_frequency_from_euler_angle(h5, PG)

    return

#------------------------------------------------------------------------------ 
# Material parameter

def sim_continuous_roll_E(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Simulate continuous roll experiments for different Young's modulus
    '''
    
    parameter = roll_parameter.get_continuous_roll_parameter()

    N, dt = 100, 0.001
    parameter['N'], parameter['dt'] = N, dt    
    E_min, E_max, E_step = 3.0, 6.01, 0.5 
    T_min, T_max = 5.0, 20.0 
              
    T_arr = linear_map_v2T(E_min, E_max, 
        T_max, T_min, E_step, 3.0, 7.0)              
                                                            
    # Parameter Grid    
    E_param = {'v_min': E_min, 'v_max': E_max, 'N': None, 
        'step': E_step, 'round': 0, 'log': True}        

    # Poisson's ratio
    c = 0.5
    G_param = E_param.copy()
    G_param['scale'] = 1 / (2 * (1 + c))

    eta_param = E_param.copy()
    eta_param['scale'] = 1e-2
    
    nu_param = G_param.copy()
    nu_param['scale'] = 1e-2
    
    T_param = {'v_arr': T_arr, 'round': 1}
                                                    
    grid_param = {('E', 'G', 'eta', 'nu', 'T') : (E_param, G_param, eta_param, nu_param, T_param)}
    
    PG = ParameterGrid(parameter, grid_param)
                        
    filename = (f'continuous_roll_E_min={E_min}_E_max={E_max}_E_step={E_step}_'
    f'c={c:.2f}_A_={parameter["A_dv"]}_f={parameter["f_dv"]}_s0_t={parameter["s0_t"]}_'
    f's0_h={parameter["s0_h"]}_s0_t={parameter["s0_t"]}_N_{N}_dt_{dt}.h5')               
    
    h5_filepath = join(data_path, 'parameter_sweeps', filename)
    exper_spec = 'CRE'
    
    h5 = sim_exp_wrapper(simulate,
        save_raw_data,       
        N_worker,
        PG,
        RollManeuverExperiment.continuous_roll,
        h5_filepath,
        log_dir,
        output_dir,
        exper_spec,
        overwrite,
        debug)
    
    # Analyse raw data and save results        
    compute_roll_frequency(h5, PG)    
    compute_roll_frequency_from_euler_angle(h5, PG)

    return

def sim_continuous_roll_E_p(N_worker, 
        simulate = True,
        save_raw_data = True,
        overwrite = False,
        debug = False):
    '''
    Simulate continuous roll experiments for different Young's modulus and
    Poisson's ratio nu
    '''

    parameter = roll_parameter.get_continuous_roll_parameter()
    
    N, dt = 100, 0.001
    parameter['N'], parameter['dt'] = N, dt    
    E_min, E_max, E_step = 3.0, 6.01, 0.25 
    T_min, T_max = 5.0, 20.0 
              
    T_arr = linear_map_v2T(E_min, E_max, 
        T_max, T_min, E_step, 3.0, 7.00)   

    # Parameter Grid    
    E_param = {'v_min': E_min, 'v_max': E_max, 'N': None, 
        'step': E_step, 'round': 0, 'log': True}        

    eta_param = E_param.copy()
    eta_param['scale'] = 1e-2
    
    T_param = {'v_arr': T_arr, 'round': 1}

    # Poisson's ratio
    p_arr = np.arange(0.1, 0.51, 0.1)

    for p in p_arr:

        G_param = E_param.copy()
        G_param['scale'] = 1 / (2 * (1 + p))
    
        nu_param = G_param.copy()
        nu_param['scale'] = 1e-2
                                                
        grid_param = {('E', 'G', 'eta', 'nu', 'T') : 
            (E_param, G_param, eta_param, nu_param, T_param)}
        
        PG = ParameterGrid(parameter, grid_param)
                        
        filename = (f'continuous_roll_E_min={E_min}_E_max={E_max}_E_step={E_step}_'
        f'p={p:.2f}_A_={parameter["A_dv"]}_f={parameter["f_dv"]}_s0_t={parameter["s0_t"]}_'
        f's0_h={parameter["s0_h"]}_s0_t={parameter["s0_t"]}_N_{N}_dt_{dt}.h5')               
    
        h5_filepath = join(data_path, 'parameter_sweeps', filename)
        exper_spec = 'CRE'
    
        h5 = sim_exp_wrapper(simulate,
            save_raw_data,       
            N_worker,
            PG,
            RollManeuverExperiment.continuous_roll,
            h5_filepath,
            log_dir,
            output_dir,
            exper_spec,
            overwrite,
            debug)
    
        # Analyse raw data and save results        
        compute_roll_frequency(h5, PG)    
        compute_roll_frequency_from_euler_angle(h5, PG)

    return

def sim_one_turn():
    
    pass

if __name__ == '__main__':
    
    N_worker = 7
    
    # sim_continuous_roll_f(N_worker, 
    #     simulate = False, save_raw_data=True, debug=False, overwrite=False)
    
    # sim_continuous_roll_A(N_worker, 
    #     simulate = True, save_raw_data=True, debug=False, overwrite=False)        
            
    # sim_continuous_roll_s0_t(N_worker, 
    #     simulate=True, save_raw_data=True, debug=False, overwrite=False)
    
    # sim_continuous_roll_A_f(N_worker, 
    #     simulate = False, save_raw_data=True, debug=False, overwrite=False)

    # sim_continuous_roll_A_f(N_worker, 
    #     simulate = False, save_raw_data=True, debug=False, overwrite=False)
    
    # sim_continuous_roll_mu(N_worker, 
    #     simulate = True, save_raw_data = True, overwrite = False, debug = False)

    # sim_continuous_roll_E(N_worker, 
    #     simulate = True, save_raw_data = True, overwrite = False, debug = False)

    sim_continuous_roll_E_p(N_worker, 
        simulate = True, save_raw_data = True, overwrite = False, debug = False)

    
    print('Finished!')






