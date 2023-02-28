'''
Created on 10 Feb 2023

@author: lukas
'''
# third-party imports
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# local imports
from forward_worm.coil.coil_dirs import log_dir, sim_dir, sweep_dir, video_dir, fig_dir
from parameter_scan import ParameterGrid
from simple_worm_experiments.worm_studio import WormStudio


def load_h5_file(filename):

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

#------------------------------------------------------------------------------ 
# Generate videos

def generate_worm_videos(h5_filename):

    h5_filepath = sweep_dir / h5_filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    vid_filenames = [f'A_dv={p["A_dv"]}_A_lr={p["A_lr"]}_'
    f'lam_dv={p["lam_dv"]}_lam_lr={p["lam_lr"]}_'
    f'f_dv={p["f_dv"]}_f_lr={p["f_lr"]}_s0_lr={p["s0_lr_t"]}_'
    f'_mu_{p["mu"]:.2f}_N_{p["N"]}_dt_{p["dt"]}' for p in PG.param_arr]

    output_dir = video_dir / Path(h5_filename).stem
    if not output_dir.exists(): output_dir.mkdir()
                                                                                             
    WormStudio.generate_worm_clips_from_PG(
        PG, vid_filenames, output_dir, sim_dir, 
        add_trajectory = False,
        draw_e3 = False,
        n_arrows = 0.2)

    return

#------------------------------------------------------------------------------ 
# Plot angle

def plot_roll_frequency_kinematic_parameter(
        h5_filepath_A, 
        h5_filepath_f,
        h5_filepath_s0_t,
        show = False):
    '''
    Plot roll frequency as a function of the kinematic frequency, 
    muscle amplitude, muscle frequency, 
    
    :param h5_filepath_A (str): 
    :param h5_filepath_f (str):
    :param h5_filepath_s0_t (str):    
    :param show (bool):
    '''
    
    h5_A, PG_A = load_h5_file(h5_filepath_A)
    h5_f, PG_f = load_h5_file(h5_filepath_f)
    h5_s0_t, PG_s0_t = load_h5_file(h5_filepath_s0_t)

    A_arr = PG_A.v_from_key('A_dv')     
    f_muscle_A = PG_A.base_parameter['f_dv']
    f_muscle_arr = PG_f.v_from_key('f_dv')
    s0_t_arr = PG_s0_t.v_from_key('s0_lr_t')
    f_muscle_s0_t = PG_s0_t.base_parameter['f_dv']

    f_arr_A = h5_A['f_R_avg'][:] 
    f_arr_f = h5_f['f_R_avg'][:]
    f_arr_s0_t = h5_s0_t['f_R_avg'][:]
        
    fig = plt.figure(figsize=(12, 8))
    
    gs = plt.GridSpec(2, 3)
    
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax02 = plt.subplot(gs[0,2])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])
    ax12 = plt.subplot(gs[1,2])
                
    ax00.plot(A_arr, f_arr_A, '-o')
    ax01.plot(f_muscle_arr, f_arr_f, '-o')
    ax02.plot(s0_t_arr, f_arr_s0_t, '-o')

    ax10.plot(A_arr, f_arr_A / f_muscle_A, '-o')
    ax11.plot(f_muscle_arr, f_arr_f / f_muscle_arr, '-o')
    ax12.plot(s0_t_arr, f_arr_s0_t / f_muscle_s0_t, '-o')
        
    lfz = 20
    
    ax00.set_ylabel('$f_\mathrm{R}$', fontsize = lfz)    
    ax10.set_ylabel('$f_\mathrm{R} / f_\mathrm{M}$', fontsize = lfz)
    ax10.set_xlabel('$A$', fontsize = lfz)
    ax11.set_xlabel('$f_\mathrm{M}$ [Hz]', fontsize = lfz)
    ax12.set_xlabel('$s_0$', fontsize = lfz)
        
    plt.tight_layout()
    
    if show: plt.show()

    filename = 'roll_frequency_A_f_s0.png'
    plt.savefig(fig_dir / filename)
    
    return

if __name__ == '__main__':
    
    
#------------------------------------------------------------------------------ 
# Plots
   
    plot_roll_frequency_kinematic_parameter(
'continuous_coil_A_min=1.0_A_max=4.01_A_step=0.25_f=5.0_s0_lr=0.25_N=100_dt=0.01.h5',        
'continuous_coil_f_min=1.0_f_max=8.01_f_step=0.5_A=4.0_s0_lr=0.25_N=100_dt=0.01.h5',
'continuous_coil_s0_lr_min=0.1_s0_lr_max=0.5_s0_lr_step=0.01_A=4.0_f=5.0_N=100_dt=0.01.h5',        
    show = True)        

#------------------------------------------------------------------------------ 
# Videos

#     generate_worm_videos(
# 'continuous_coil_A_min=1.0_A_max=4.01_A_step=1.0_f=5.0_s0_lr=0.25_N=100_dt=0.01.h5')

#     generate_worm_videos(
# 'continuous_coil_s0_lr_min=0.1_s0_lr_max=0.5_s0_lr_step=0.05_A=4.0_f=5.0_N=100_dt=0.01.h5')

        
    print('Finished')
