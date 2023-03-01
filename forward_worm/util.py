'''
Created on 17 Aug 2022

@author: lukas
'''

# Build-in imports
from os.path import join
import numpy as np
import h5py
from argparse import ArgumentParser

# Local imports
from simple_worm_experiments.grid_loader import GridLoader
from simple_worm_experiments.experiment import wrap_simulate_experiment

from mp_progress_logger.custom_loggers import FWProgressLogger

def default_sweep_parameter():
    '''
    Default sweep hyper parameter
    '''            
    parser = ArgumentParser(description = 'sweep-parameter')
    
    parser.add_argument('--N_worker', type = int, default = 10,
        help = 'Number of processes') 
    parser.add_argument('--simulate', type = bool, default = True,
        help = 'If true, simulations are run from scratch') 
    parser.add_argument('--save_raw_data', type = bool, default = True,
        help = 'If true, FrameSequences are pickled to disk') 
    parser.add_argument('--overwrite', type = bool, default = False,
        help = 'If true, already existing simulation results are overwritten')
    parser.add_argument('--debug', type = bool, default = False,
        help = 'If true, exception handling is turned off which is helpful for debugging')    
    
    return parser

def simulate_experiments(N_worker, 
        PG, 
        task,                    
        log_dir,
        output_dir,
        overwrite = False,
        exper_spec = '',
        debug = False):
    
    '''
    Runs the experiment defined by the task function for all parameters in 
    ParameterGrid 
        
    :param N_worker (int): Number of processes
    :param PG (ParameterGrid): Parameter grid
    :param task (funtion): function
    :param log_dir (str): log file directory
    :param output_dir (str): output directory
    :param overwrite (boolean): If true, existing files are overwritten
    :param exper_spec (str): experiment descriptor
    :param debug (boolean): Set to true, to debug 
    '''
        
    # Creater status logger for experiment
    # The logger will log and display
    # the progress and outcome of the simulations
    PGL = FWProgressLogger(PG, 
                       log_dir, 
                       pbar_to_file = False,                        
                       pbar_path = './pbar/pbar.txt', 
                       exper_spec = exper_spec,
                       debug = debug)

    # Start experiments
    PGL.run_pool(N_worker, 
                 wrap_simulate_experiment, 
                 task,
                 output_dir,                 
                 overwrite = overwrite)
    
    return PGL

def save_experiment_to_h5(PG,                
        h5_filepath,
        output_dir,
        log_dir, 
        FS_keys = ['x', 'Omega'], 
        CS_keys = 'Omega'):    
    
    '''
    Pools experiment results and saves them to single HDF5
    
    :param PG (ParameterGrid): Parameter grid object
    :param h5_filepath (str): HDF5 filepath  
    :param output_dir (str): Output directory FrameSequence
    :param h5_filepath (str): Log file directory    
    :param FS_keys (list): Frame variables to save to h5
    :param CS_keys (list): Control variables to save to h5
    '''    
    
    # Save results to HDF5            
    grid_filename = PG.filename + '.json'
    grid_param_path = join(log_dir, grid_filename)    
    
    GP = GridLoader(grid_param_path, output_dir)            
        
    h5 = GP.save_data(h5_filepath, FS_keys, CS_keys)            

    h5['PG'] = PG.filename

    exit_status = h5['exit_status'][:]

    print(f'Finished simulations: {np.sum(exit_status)/len(exit_status)*100}% failed')    
    print(f'Saved parameter scan simulations results to {h5_filepath}')
    
    return h5

def sim_exp_wrapper(simulate, 
        save_raw_data,       
        FS_keys,
        CS_keys,
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
            PG, CS_func, 
            str(log_dir), str(output_dir),
            exper_spec = exper_spec,
            overwrite = overwrite,
            debug = debug)

        PGL.close()
                
    if save_raw_data: # Save raw simulation results to h5                                     
        h5 = save_experiment_to_h5(PG, 
            h5_filepath,
            str(output_dir), str(log_dir), 
            FS_keys = FS_keys, 
            CS_keys = CS_keys)
    
    else:        
        assert not simulate, 'If simulate is True, save_raw_data must be True'
        h5 = h5py.File(h5_filepath, 'r+')    
    
    return h5

