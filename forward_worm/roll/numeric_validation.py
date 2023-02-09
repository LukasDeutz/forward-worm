'''
Created on 11 Jan 2023

@author: lukas
'''

#Built-in imports
from os.path import join

#Third-party imports
import numpy as np

# Local imports
import forward_worm.roll.roll_parameter as roll_parameter
from forward_worm.roll.roll_parameter import data_path, log_dir, output_dir
from forward_worm.util import simulate_experiments, save_experiment_to_h5
from simple_worm_experiments.roll.roll import RollManeuverExperiment
from simple_worm_experiments.grid_loader import GridLoader, GridPoolLoader
from parameter_scan import ParameterGrid
from parameter_scan.util import load_grid_param, load_file_grid
from mp_progress_logger import FWProgressLogger 
    
#------------------------------------------------------------------------------ 
# Numerical validation
    
def sim_dt(N_worker):
    
    parameter = roll_parameter.get_repetitious_roll_parameter()
        
    # Decrease time step by powers of two                
    e = np.arange(0, 10)
    dt_arr = (0.01 / 2**e).tolist()   
        
    # Create parameter grid 
    dt_param = {'v_arr': dt_arr, 'round': None}                
    grid_param = {'dt': dt_param}
    PG = ParameterGrid(parameter, grid_param)
        
    PGL = simulate_experiments(N_worker, 
                           PG, 
                           RollManeuverExperiment.repetitious_roll,
                           log_dir,
                           output_dir)
        
    # Save rare simulation results to h5
    h5_filepath = join(data_path, 'parameter_sweeps', f'repetitious_roll_dt.h5')
     
    save_experiment_to_h5(PG, 
                      h5_filepath,
                      output_dir,
                      log_dir, 
                      FS_keys = ['x', 'e1', 'e2', 'e3'], 
                      CS_keys = None)

    # Close logger
    PGL.close()
                                                                                
    return

def sim_N():

    parameter = roll_parameter.get_repetitious_roll_parameter()
        
    # Decrease time step by powers of two                
    e = np.arange(0, 5)
    N_arr = (int(100 * 2**e)).tolist()   
        
    # Create parameter grid 
    N_param = {'v_arr': N_arr, 'round': None}                
    grid_param = {'dt': N_param}
    PG = ParameterGrid(parameter, grid_param)
        
    PGL = simulate_experiments(N_worker, 
                           PG, 
                           RollManeuverExperiment.repetitious_roll,
                           log_dir,
                           output_dir)
        
    # Save rare simulation results to h5
    h5_filepath = join(data_path, 'parameter_sweeps', f'repetitious_roll_dt.h5')
     
    save_experiment_to_h5(PG, 
                      h5_filepath,
                      output_dir,
                      log_dir, 
                      FS_keys = ['x', 'e1', 'e2', 'e3'], 
                      CS_keys = None)

    # Close logger
    PGL.close()
    
    return


if __name__ == '__main__':
    
    N_worker = 2    
    sim_dt(N_worker)
    sim_N(N_worker)

    
    

