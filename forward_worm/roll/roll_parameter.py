'''
Created on 11 Jan 2023

@author: lukas

Parameter

Default parameter for roll experiments
'''
import numpy as np
import pickle
from pathlib import Path

data_path = Path('../../data/roll')
log_dir = data_path / 'logs'
output_dir = data_path / 'simulations'
fig_path = Path('../../figures/roll')
video_path = Path('../../videos/roll')
animation_path = Path('../../animations')

def base_parameter():
    
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
    
    # Fluid parameter
    parameter['external_force'] = ['rft']    
    parameter['rft'] = 'Whang'
    parameter['mu'] = 1e-3
                
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
    parameter['smo'] = False
    parameter['Ds'] = 1.0/150
    parameter['s0'] = False
    
    parameter['mts'] = True
    parameter['tau_on'] = 0.05
    parameter['Dt_on'] = 3*parameter['tau_on']
        
    return parameter

def get_continuous_roll_parameter(save = False):
    
    parameter = base_parameter()
    
    # Kinematic parameter
    parameter['A_dv'] = 6.0
    parameter['f_dv'] = 5.0    

    parameter['A_lr'] = 6.0
    parameter['f_lr'] = 5.0    
    parameter['phi'] = np.pi/2

    parameter['Ds_h'] = None
    parameter['s0_h'] = None

    parameter['Ds_t'] = 1.0/32
    parameter['s0_t'] = 0.25
    
    if save:
        pickle.dump(parameter, open('./continuous_roll_parameter.dat', 'wb'))    
                
    return parameter

if __name__ == '__main__':
    
    get_continuous_roll_parameter(save = True)
    
    print('Finished!')
    
    

    
    
    
    
    

