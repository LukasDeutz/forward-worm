'''
Created on 12 Jan 2023

@author: lukas
'''
# Build-in imports
from os.path import join
from os import mkdir

# Third-party imports
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation 
import numpy as np
from pathlib import Path
import pickle

# Local imports
from forward_worm.roll.roll_parameter import data_path, fig_path, video_path, animation_path
from parameter_scan import ParameterGrid
from simple_worm_experiments.worm_studio import WormStudio
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_multiple_scalar_fields

def load_h5_file(filename):

    h5_filepath = data_path / 'parameter_sweeps' / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = data_path / 'logs' / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

def generate_worm_videos(N_worker, h5_filepath):

    h5, PG = load_h5_file(h5_filepath)
                        
    FS_filenames = [h + '.dat' for h in PG.hash_arr]
       
    vid_filenames = [f'A={p["A_dv"]}_f={p["f_dv"]}_s0_h_{p["s0_h"]}_s0_t_{p["s0_t"]}'
        f'_mu_{p["mu"]:.3f}_N_{p["N"]}_dt_{p["dt"]}' for p in PG.param_arr]

    video_dir = video_path / Path(h5_filepath).stem
    
    if not video_dir.exists(): 
        video_dir.mkdir()
     
    WormStudio.generate_clips_in_parallel(N_worker,
        FS_filenames,
        vid_filenames,
        data_path,
        video_dir,
        add_trajectory = False,
        draw_e3 = False,
        n_arrows = 0.2)
                     
    return
    
def plot_continuous_roll_frequency(h5_filepath_A_f, show = False):
    '''
    Plot roll frequency as a function of the muscle amplitude 
    for different muscle frequencies
    
    :param h5_filepath_A_f (str): hdf5 filepath
    :param show: If true, show plot
    '''
    
    h5, PG = load_h5_file(h5_filepath_A_f)
        
    A_arr = PG.v_from_key('A_dv')    
    f_muscle_arr = PG.v_from_key('f_dv')
    
    fig = plt.figure(figsize=(6, 8))
    
    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    for i in range(PG.shape[1]):
        idx_mat = PG[:, i]        
        idx_arr = PG.flat_index(idx_mat)
        
        f_avg = h5['f_avg'][idx_arr]        
        ax0.plot(A_arr, f_avg, '-o', label = f'$f_M = {f_muscle_arr[i]}$')                
        ax1.plot(A_arr, f_avg / f_muscle_arr[i], '-o', label = f'$f_M = {f_muscle_arr[i]}$')

    lfz = 20
    legfz = 12
    ax0.legend(fontsize = legfz)
    ax0.set_ylabel('$f_\mathrm{R}$ [Hz]', fontsize = lfz)
    ax1.set_ylabel('$f_\mathrm{R} / f_\mathrm{M}$', fontsize = lfz)
    ax1.set_xlabel('$A$', fontsize = lfz)

    plt.tight_layout()
    
    if show: plt.show()

    filename = 'roll_frequency_A.png'
    plt.savefig(fig_path / filename)
                
    return

def plot_continuous_roll_frequency_kinematic_parameter(
        h5_filepath_A, 
        h5_filepath_f,
        h5_filepath_s0_t,
        h5_filepath_mu,        
        N_phi = 4,
        show = False):
    '''
    Plot roll frequency as a function of the muscle amplitude and
    as a function of the muscle frequency.
    
    :param h5_filepath_A: 
    :param h5_filepath_f:
    :param h5_filepath_s0_t:    
    :param N_phi:
    :param show:
    '''
    
    h5_A, PG_A = load_h5_file(h5_filepath_A)
    h5_f, PG_f = load_h5_file(h5_filepath_f)
    h5_s0_t, PG_s0_t = load_h5_file(h5_filepath_s0_t)
    h5_mu, PG_mu = load_h5_file(h5_filepath_mu)

    A_arr = PG_A.v_from_key('A_dv') 
    f_muscle_A = PG_A.base_parameter['f_dv']
    f_muscle_arr = PG_f.v_from_key('f_dv')
    s0_t_arr = PG_s0_t.v_from_key('s0_t')
    f_muscle_s0_t = PG_s0_t.base_parameter['f_dv']
    mu_arr = PG_mu.v_from_key('mu')
    f_muscle_mu = PG_mu.base_parameter['f_dv']

    f_arr_A = h5_A['f_avg_euler'][:] 
    f_arr_f = h5_f['f_avg_euler'][:]
    f_arr_s0_t = h5_s0_t['f_avg_euler'][:]
    f_arr_mu = h5_mu['f_avg_euler'][:]
        
    fig = plt.figure(figsize=(12, 8))
    
    gs = plt.GridSpec(2, 4)
    
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax02 = plt.subplot(gs[0,2])
    ax03 = plt.subplot(gs[0,3])    
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])
    ax12 = plt.subplot(gs[1,2])
    ax13 = plt.subplot(gs[1,3])
                
    ax00.plot(A_arr, f_arr_A, '-o')
    ax01.plot(f_muscle_arr, f_arr_f, '-o')
    ax02.plot(s0_t_arr, f_arr_s0_t, '-o')
    ax03.semilogx(mu_arr, f_arr_mu, '-o')

    ax10.plot(A_arr, f_arr_A / f_muscle_A, '-o')
    ax11.plot(f_muscle_arr, f_arr_f / f_muscle_arr, '-o')
    ax12.plot(s0_t_arr, f_arr_s0_t / f_muscle_s0_t, '-o')
    ax13.semilogx(mu_arr, f_arr_mu / f_muscle_mu, '-o')
        
    lfz = 20
    
    ax00.set_ylabel('$f_\mathrm{R}$', fontsize = lfz)    
    ax10.set_ylabel('$f_\mathrm{R} / f_\mathrm{M}$', fontsize = lfz)
    ax10.set_xlabel('$A$', fontsize = lfz)
    ax11.set_xlabel('$f_\mathrm{M}$ [Hz]', fontsize = lfz)
    ax12.set_xlabel('$s_0$', fontsize = lfz)
    ax13.set_xlabel('$\mu$', fontsize = lfz)
        
    plt.tight_layout()
    
    if show: plt.show()

    filename = 'roll_frequency_A_f_s0.png'
    plt.savefig(fig_path / filename)
    
    return

def plot_continuous_roll_frequency_material_parameter(
        h5_filename_list,
        nu_arr,
        show = False):

    h5, PG = load_h5_file(h5_filename_list[0])
    
    E_arr = PG.v_from_key('E')         
    f_muscle_E = PG.base_parameter['f_dv']    
                    
    plt.figure(figsize = (12, 8))
    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    for h5_filename, nu in zip(h5_filename_list, nu_arr):
        
        h5, PG = load_h5_file(h5_filename)

        f_arr = h5['f_avg_euler'][:]
        
        f_arr[np.argwhere(np.isnan(f_arr))] = 0.0
            
        ax0.semilogx(E_arr, f_arr, '-o', label = fr'$\nu={nu:.2f}$')
        ax1.semilogx(E_arr, f_arr / f_muscle_E, '-o')

        # ax0.loglog(E_arr, f_arr, '-o', label = fr'$\nu={nu:.2f}$')
        # ax1.semilogx(E_arr, f_arr / f_muscle_E, '-o')

    
    fz = 20
    ax0.set_ylabel(r'$f_\mathrm{R}$', fontsize = 20)    
    ax1.set_ylabel(r'$f_\mathrm{R} / f_\mathrm{M}$', fontsize = 20)
    ax1.set_xlabel(r'$E$', fontsize = 20)
    ax0.legend(fontsize = 12)
    
    if show: plt.show()
    
    #Save
        
    return

        
        
def plot_continuous_roll_angle_and_roll_frequency(h5_filepath, 
        key, 
        normalize = True,
        show = False,
        N_phi = 3):
    '''
    Plot rotation angle and roll frquency
    
    :param h5_filepath (str): h5py filepath
    :param key (str): kinetmatic parameter
    :param normalize (bool): If True, roll frequency is noramlized by muscle frequency
    :param show (bool): If true, plot is shown 
    :param N_phi (int): Plot rotation angle for N_phi experiments
    '''
        
    h5, PG, = load_h5_file(h5_filepath)
    
    v_arr = PG.v_from_key(key)
    
    alpha_list = [h5['alpha'][h][:] for h in PG.hash_arr]    
    t_list = [h5['t'][h][:] for h in PG.hash_arr] 
    
    # Roll angle over time is only plotted of N_phi experiments
    # Otherwise the plot get's to croweded
    alpha_idx_arr = np.linspace(0, len(v_arr) - 1, N_phi).astype(int)

    if key != 'A_dv':
        f_muscle = PG.base_parameter['f_dv'] 
                
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(3,1)
    ax0 = plt.subplot(gs[0])        
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
            
    # Create norm
    cm = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=np.min(v_arr), vmax=np.max(v_arr))    
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    
    f_avg_arr = h5['f_avg'][:]
    f_avg_euler_arr = h5['f_avg_euler'][:]
    
    if normalize:
        if key in ['f_dv', 'f_lr']: 
            f_muscle_arr = v_arr
        else: 
            f_muscle_arr = np.array(len(v_arr) * [PG.base_parameter['f_dv']])
                    
    colors = sm.to_rgba(v_arr)

    for n in alpha_idx_arr: 
        alpha, t, c = alpha_list[n], t_list[n], colors[n] 
        alpha_avg = alpha.mean(axis = 1)
        alpha_avg = alpha_avg % (2*np.pi) - np.pi
        alpha_avg = 180*alpha_avg / np.pi
    
        ax0.plot(t, alpha_avg, c = c)
    
    ax1.plot(v_arr, f_avg_arr, '-', c = 'k')
    ax1.plot(v_arr, f_avg_euler_arr, '--', c = 'k')
    
    ax2.plot(v_arr, f_avg_arr / f_muscle_arr, '-', c = 'k')
    ax2.plot(v_arr, f_avg_euler_arr / f_muscle_arr, '--', c = 'k')
        
    for v, f_avg, f_avg_euler, f_muscle, c in zip(v_arr, 
        f_avg_arr, f_avg_euler_arr, f_muscle_arr, colors):
                
        ax1.plot(v, f_avg, 'o', c = c)
        ax1.plot(v, f_avg_euler, 'v', c = c)

        ax2.plot(v, f_avg / f_muscle, 'o', c = c)
        ax2.plot(v, f_avg_euler / f_muscle, 'v', c = c)

    ax0.set_xlabel(r'$t$ [s]', fontsize = 20)       
    ax0.set_ylabel(r'$\alpha$ [degree]', fontsize = 20)
       
    if key == 'A_dv':
        xl = r'$A$'
    elif key == 'f_dv':
        xl = r'$f_\mathrm{M}$'
    elif key == 's0_t':
        xl = r'$s_\mathrm{Neck}$'
       
    ax2.set_xlabel(xl, fontsize = 20)       
    
    ax1.set_ylabel(r'$f_\mathrm{R}$ [Hz]', fontsize = 20)   
    ax2.set_ylabel(r'$f_\mathrm{R} / f_\mathrm{M}$ [Hz]', fontsize = 20)       
        
    cb = plt.colorbar(sm, ax = [ax0, ax1, ax2])
    cb.set_label(xl, fontsize = 20)
    
    if show: plt.show()

    #plt.tight_layout()
   
    filename = f'roll_frequency_{key}_f_and_phi.png'
    plt.savefig(fig_path / filename)
       
    return 

#------------------------------------------------------------------------------ 

def plot_controls_and_strains(h5_filepath):
    '''
    Plot controls and strains for all simulations associated with 
    the given h5py file
    
    :param h5_filepath (str): h5py filepath
    :param key (str): kinetmatic parameter
    '''

    h5, PG = load_h5_file(h5_filepath)
    
    if PG.base_parameter['dt_report'] is not None:
        dt = PG.base_parameter['dt_report']
    else:
        dt = PG.base_parameter['dt']
        
    sim_filepaths = [(data_path / 'simulations' / h).with_suffix('.dat') for h in PG.hash_arr]

    fig_dir = fig_path / Path(h5_filepath).stem / 'controls_vs_strains'
    
    if not fig_dir.exists(): fig_dir.mkdir(parents=True, exist_ok=True)

    fig_filenames = [f'A={p["A_dv"]}_f={p["f_dv"]}_s0_h_{p["s0_h"]}_s0_t_{p["s0_t"]}'
        f'_mu_{p["mu"]:.3f}_N_{p["N"]}_dt_{p["dt"]}' for p in PG.param_arr]
                
    for sim_filepath, fig_filename, param in zip(sim_filepaths, fig_filenames, PG.param_arr):
                
        T_max = 8.0 / param['f_dv']

        data = pickle.load(open(sim_filepath, 'rb'))
        FS, CS = data['FS'], data['CS']
        fig = plot_CS_vs_FS(CS, FS, T_max = T_max, T = T_max)                        
        plt.savefig(str((fig_dir / fig_filename).with_suffix('.png')))
        plt.close(fig)
        
    return
    
def plot_roll_angle_twist_angular_velocity_chemogramm(h5_filepath, T_max = ('muscle', 4)):    
    '''
    Plot roll angle, twist and angular velocity
    
    :param h5_filepath (str): h5py filepath
    :param h5_filepath (Tuple[str, int]): First item is the number of cycles to plot, if second item
        is "muscle" then muscle period is used, if its "roll" roll period is used             
    '''

    h5, PG = load_h5_file(h5_filepath)
            
    fig_dir = fig_path / Path(h5_filepath).stem / ('roll_twist_velocity_' + T_max[0] + '_resolution')
    
    if not fig_dir.exists(): fig_dir.mkdir(parents=True, exist_ok=True)

    fig_filenames = [f'A={p["A_dv"]}_f={p["f_dv"]}_s0_h_{p["s0_h"]}_s0_t_{p["s0_t"]}'
        f'_mu_{p["mu"]:.3f}_N_{p["N"]}_dt_{p["dt"]}' for p in PG.param_arr]
                                                  
    f_avg_euler_arr = h5['f_avg_euler'][:]
        
    for h, fig_filename, param, f_avg_euler in zip(PG.hash_arr, fig_filenames, PG.param_arr, f_avg_euler_arr):
        
        # load roll, twist and angular velocity
        alpha = h5['alpha'][h][:]
        k3 = h5['FS']['Omega'][h][:, 2, :]
        w = h5['FS']['w'][h][:, 2, :]
                
        # crop data to n muscle or roll cycles                                
        if T_max[0] == 'muscle':
            t_max = T_max[1] / param['f_dv']            
        elif T_max[0] == 'roll': 
            t_max = T_max[1] / f_avg_euler            
        else:
            assert False, 'First item of T_max must be a string in ["muscle", "roll"]'
        
        if t_max > param['T']: t_max = param['T']
        
        idx_arr = h5['t'][h][:] <= t_max
        
        alpha = alpha[idx_arr, :]
        k3 = k3[idx_arr, :]
        w = w[idx_arr, :]
        
        # map range of roll angle to [-180, 180]
        alpha = alpha % (2 * np.pi) - np.pi
        alpha = 180*alpha / np.pi
                                                         
        fig = plot_multiple_scalar_fields([alpha, k3, w],
            T = t_max,
            titles = [r'$\alpha$', r'$\kappa_3$', r'$\omega_3$'],
            cmaps = [plt.cm.twilight, plt.cm.PRGn, plt.cm.PiYG],
            fig_size = (12, 3*6),
            grid_layout = (3,1)
        )
                                        
        fp = fig_dir / Path(fig_filename).with_suffix('.png')        
        plt.savefig(str(fp))
        plt.close(fig)
        
    return

def plot_body_points_twist_angular_velocity(
        h5_filepath, 
        s_points = [0.1, 0.5],
        T_max = ('muscle', 4)):
    '''
    Plot roll angle, twist and angular velocity
    
    :param h5_filepath (str): h5py filepath
    :param s_points (List[floats]): body points to plot        
    '''

    h5, PG = load_h5_file(h5_filepath)
            
    fig_dir = fig_path / Path(h5_filepath).stem / ('body_points_twist_velocity' + T_max[0] + '_resolution')
    
    if not fig_dir.exists(): fig_dir.mkdir(parents=True, exist_ok=True)

    fig_filenames = [f'A={p["A_dv"]}_f={p["f_dv"]}_N_{p["N"]}_dt_{p["dt"]}'
        f'_s0_h_{p["s0_h"]}_s0_t_{p["s0_t"]}' for p in PG.param_arr]
                                                  
    f_avg_euler_arr = h5['f_avg_euler'][:]
            
    s_arr = np.linspace(0, 1, PG.base_parameter['N'])                
    s_idx = np.array([np.abs(s_arr - s).argmin() for s in s_points])
                    
    for h, fig_filename, param, f_avg_euler in zip(PG.hash_arr, fig_filenames, PG.param_arr, f_avg_euler_arr):
        
        
        # crop data to n muscle or roll cycles                                
        if T_max[0] == 'muscle':
            t_max = T_max[1] / param['f_dv']            
        elif T_max[0] == 'roll': 
            t_max = T_max[1] / f_avg_euler            
        else:
            assert False, 'First item of T_max must be a string in ["muscle", "roll"]'
        
        if t_max > param['T']: t_max = param['T']
                
        idx_arr = h5['t'][h][:] <= t_max
        t = h5['t'][h][idx_arr]                      
        
        # load roll, twist and angular velocity
        k3 = h5['FS']['Omega'][h][:, 2, :]
        w = h5['FS']['w'][h][:, 2, :]
        # get selected body points and time window
        w = w[idx_arr, :][:, s_idx]        
        k3 = k3[idx_arr, :][:, s_idx]
                
        S = len(s_points)
        
        fig = plt.figure()
        gs = plt.GridSpec(S , 1)
        
        axes = [plt.subplot(gs[i]) for i in range(S)]

        for i, s in enumerate(s_points):
            axes[i].plot(t, k3[:, i], c = 'b')
            axes[i].set_ylabel(r'$\kappa_3$', fontsize = 20)            
            ax_twin = axes[i].twinx()
            ax_twin.plot(t, w[:, i], c = 'r')
            ax_twin.set_ylabel(r'$\omega_3$', fontsize = 20)
            axes[i].set_xlabel(r'$t$', fontsize = 20)
                                        
        fp = fig_dir / Path(fig_filename).with_suffix('.png')        
        plt.savefig(str(fp))
        plt.close(fig)


#------------------------------------------------------------------------------ 
# Animate head trajectories

def animate_single_2D_point_trajectory(x, 
        y,
        t = None,
        L = 10,
        interval = 50.0,
        figsize = (8,8),
        c = 'red'):
    '''
    Animates centrelinecoordinates of the head 
    projected onto the principle plane.
    
    n = number of time steps
        
    :param x (np.ndarray n x 1): x coordinates
    :param y (np.ndarray n x 1): y coordinates   
    :param t (np.ndarray n x 1): time steps, defaults to None   
    :param L (int): Number of lines
    :param interval (float): time interval between frames in ms
    :param figsize (tuple): defaults to (8,8)
    :param c (str): default to 'red'            
    '''
            
    N = len(x) # Number of frames
    
    fig = plt.figure(figsize = figsize)
    ax = plt.subplot(111)
    
    lines = [ax.plot([], [], '-', c = c, lw = 2.5, ms = 0.0)[0] for _ in range(L)]
    alpha = np.linspace(0, 1, L)
    
    marker, = ax.plot([], [], 'o', c = c, ms = 10.0) 
    time_text = ax.text(0.85, 0.95, s='', 
        fontsize = 15, transform = ax.transAxes)
        
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    scale = 0.05 
    x_min -= scale*np.abs(x_min)
    x_max += scale*np.abs(x_max)
    y_min -= scale*np.abs(y_min)
    y_max += scale*np.abs(y_max)
    
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect('equal')                        
       
    ax.set_xlabel('$x$', fontsize = 20)
    ax.set_ylabel('$y$', fontsize = 20)
                                
    def update(n):

        start = n-L
        if start < 0: start = 0
                                
        for i, j in enumerate(range(start, n)):
            lines[i].set_data([x[j], x[j+1]], 
                [y[j], y[j+1]])
            lines[i].set_alpha(alpha[i])

        marker.set_data(x[n], y[n])
        
        if t is not None:
            time_text.set_text(f't={t[n]:.2f}')
                
        return lines + [marker, time_text]

                                                   
    # calling the animation function    
    anim = animation.FuncAnimation(fig, 
        update,
        frames = N,
        interval = interval,
        blit = True)    
                
    return anim
    

def animate_multiple_2D_point_trajectories(X, 
        Y,
        t = None,
        c_arr = None,
        L = 10,
        interval = 50.0,
        figsize = (8,8),
        cmap = 'cool',
        cbar_label = ''):
    '''
    Animates centrelinecoordinates of the head 
    projected onto the principle plane.
    
    n = number of time steps
        
    :param X (List[np.ndarray n x 1]): x coordinates
    :param Y (List[np.ndarray n x 1]): y coordinates   
    :param t (np.ndarray n x 1): time steps, defaults to None   
    :param L (int): Number of lines
    :param interval (float): time interval between frames in ms
    :param figsize (tuple): defaults to (8,8)
    :param c (str): default to 'red'            
    '''

    M = len(X) # Number of trajectories              
    N = len(X[0]) # Number of frames
    
    # Get colors from trajectories
    cm = plt.cm.get_cmap(cmap)        
    
    if c_arr is None:    
        c_arr = cm(np.linspace(0, 1, M))
    
    norm = mpl.colors.Normalize(c_arr.min(), c_arr.max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    colors = sm.to_rgba(c_arr)
    
    fig = plt.figure(figsize = figsize)
    ax = plt.subplot(111)
        
    all_lines = []

    for c in colors:        
        lines = [ax.plot([], [], '-', c = c, lw = 2.5, ms = 0.0)[0] for _ in range(L)]
        all_lines.append(lines)
                
    alpha = np.linspace(0, 1, L)    
    markers = [ax.plot([], [], 'o', c = c, ms = 10.0)[0] for c in colors] 
        
    time_text = ax.text(0.8, 0.9, s='', 
        fontsize = 15, transform = ax.transAxes)
      
    x_max = np.array([x.max() for x in X]).max()
    x_min = np.array([x.min() for x in X]).min()

    y_max = np.array([y.max() for y in Y]).max()
    y_min = np.array([y.min() for y in Y]).min()
        
    scale = 0.05 
    x_min -= scale*np.abs(x_min)
    x_max += scale*np.abs(x_max)
    y_min -= scale*np.abs(y_min)
    y_max += scale*np.abs(y_max)
    
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect('equal')                        
       
    ax.set_xlabel('$x$', fontsize = 20)
    ax.set_ylabel('$y$', fontsize = 20)

    fig.colorbar(sm, ax = ax, label = cbar_label)
                                    
    def update(n):

        start = n-L
        if start < 0: start = 0

        for m in range(M):            
            x = X[m]
            y = Y[m]
            
            lines = all_lines[m]            
            for i, j in enumerate(range(start, n)):
                lines[i].set_data([x[j], x[j+1]], 
                    [y[j], y[j+1]])
                lines[i].set_alpha(alpha[i])
            
            markers[m].set_data(x[n], y[n])
        
        if t is not None:
            time_text.set_text(f't={t[n]:.2f}')
         
        artists = [line for lines in all_lines for line in lines] + markers + [time_text]         
                        
        return artists

                                                   
    # calling the animation function    
    anim = animation.FuncAnimation(fig, 
        update,
        frames = N,
        interval = interval,
        blit = True)    
                
    return anim
    
    
def animate_head_trajectory(h5_filepath,
        T_max = None,
        L = 10):
    '''
    Animates a head trajectory for every simulation in
    the given h5py file.
    
    :param h5_filepath (str): h5py filepath
    '''

    h5, PG = load_h5_file(h5_filepath)

    dt = PG.base_parameter['dt']
    fps = int(1.0 / dt)    
    
    if T_max is None: n_max = None
    else: n_max = int(T_max/dt)
        
    ani_dir = animation_path / 'roll' / Path(h5_filepath).stem / 'head_trajectory' 
    
    if not ani_dir.exists(): 
        ani_dir.mkdir(parents = True)
    
    filenames = [f'A={p["A_dv"]}_f={p["f_dv"]}_s0_t_{p["s0_t"]}_N_{p["N"]}_dt_{p["dt"]}.mp4' 
        for p in PG.param_arr]
    
    for i, (h, fn) in enumerate(zip(PG.hash_arr, filenames)):
        
        # Load data
        if n_max is not None:            
            # Crop data
            if n_max < h5['FS']['x'][h].shape[0]:
                X = h5['FS']['x'][h][:n_max]
                t = h5['t'][h][:n_max]        
        else:        
            X = h5['FS']['x'][h][:]
            t = h5['t'][h][:]                     
        
        # Project coordinates onto principal plane
        
        if i == 0:
            x, y, w2_ref, w3_ref = EPP.project_into_pca_plane(X, 
                output_w = True)        
        else:        
            x, y = EPP.project_into_pca_plane(X, 
                w2_ref = w2_ref, w3_ref = w3_ref)         
                
        # Select head point
        x_head, y_head  = x[:, 0], y[:, 0]

        print('Start rendering animation')

        anim = animate_single_2D_point_trajectory(x_head, 
            y_head,
            t = t,
            L = L,
            interval = 1e3*dt)
                                        
        filepath = ani_dir / fn
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(str(filepath), writer=writervideo)
                            
        print(f'Saved animation to {str(filepath)}: {len(PG)-i-1} more to go!')
        
    return

def animate_head_trajectories(h5_filepath,
        key,
        T_max = None,
        L = 10,
        centre = False):
    '''
    Animates a head trajectory for every simulation in
    the given h5py file.
    
    :param h5_filepath (str): h5py filepath
    '''

    h5, PG = load_h5_file(h5_filepath)

    v_arr = PG.v_from_key(key)

    dt = PG.base_parameter['dt']
    fps = int(1.0 / dt)    
    
    if T_max is None: n_max = -1
    else: n_max = int(T_max/dt)
            
    x_head_list = []
    y_head_list = []
    
    h = PG.hash_arr[int(len(PG)/2)]
    X = h5['FS']['x'][h][:n_max]
    lam, w, x_avg = EPP.centreline_pca(X)
    
    w2_ref = w[:, 1]
    w3_ref = w[:, 2]
    
    for i, h in enumerate(PG.hash_arr):
        
        X = h5['FS']['x'][h][:n_max]
        # Project coordinates onto principal plane

        x, y = EPP.project_into_pca_plane(X, 
            w2_ref = w2_ref, w3_ref = w3_ref)         
        
        # Select head point
        x_head, y_head  = x[:, 0], y[:, 0]
        
        if centre:
            x_head -= x_head.mean()
            y_head -= y_head.mean()
                    
        x_head_list.append(x_head)
        y_head_list.append(y_head)
        
    t = h5['t'][h][:n_max]        

    print('Start rendering animation')
    
    anim = animate_multiple_2D_point_trajectories(
        x_head_list, 
        y_head_list,
        t = t,
        c_arr = v_arr,
        L = 10,
        interval = 1e3*dt,
        figsize = (8,8),
        cmap = 'cool',
        cbar_label = key)
        
    ani_dir = animation_path / 'roll'  
                
    if not ani_dir.exists(): 
        ani_dir.mkdir(parents = True)
    
    filepath = (ani_dir / Path(h5_filepath).stem).with_suffix('.mp4')
    writervideo = animation.FFMpegWriter(fps=fps)
    anim.save(str(filepath), writer=writervideo)
        
    print(f'Saved animation to {str(filepath)}')
        
    return


if __name__ == '__main__':
    
#------------------------------------------------------------------------------ 
# Plot figures
        
#     plot_continuous_roll_frequency(
# 'continuous_roll_A_f_A_min=1.0_A_max=7.0_A_step=1.0_f_min=2.0_f_max=8.0_f_step=2_N_100_dt_0.01_s0_h_None_s0_t_None.h5')

#     plot_continuous_roll_frequency_2(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
# 'continuous_roll_s0_t_min=0.1_s0_t_max=0.3_s0_t_step=0.01_A=5.0_f=5.0_N=100_dt=0.01.h5',
# 'continuous_roll_mu_min=-3.0_mu_max=0.0_N_mu=10_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.01.h5',
#     show = True)

#     plot_continuous_roll_frequency_material_parameter(
# 'continuous_roll_E_min=3.0_E_max=6.51_E_step=0.25_nu=0.50_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.001.h5',
#         show = True)

    nu_arr = np.arange(0.1, 0.51, 0.1)

    h5_filename_list = [
f'continuous_roll_E_min=3.0_E_max=6.01_E_step=0.5_nu={nu:.2f}_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.001.h5'
        for nu in nu_arr]

    plot_continuous_roll_frequency_material_parameter(
        h5_filename_list, nu_arr, show = True)

#------------------------------------------------------------------------------ 
# Sweeps

#     plot_continuous_roll_angle_and_roll_frequency(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         key = 'A_dv') 

#     plot_continuous_roll_angle_and_roll_frequency(
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         key = 'f_dv', show = True) 

#     plot_continuous_roll_angle_and_roll_frequency(
# 'continuous_roll_s0_t_min=0.1_s0_t_max=0.3_s0_t_step=0.01_A=5.0_f=5.0_N=100_dt=0.01.h5',
#         key = 's0_t') 

#     plot_controls_and_strains(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5')

#     plot_controls_and_strains(
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5')

#     plot_controls_and_strains(
# 'continuous_roll_s0_t_min=0.1_s0_t_max=0.3_s0_t_step=0.01_A=5.0_f=5.0_N=100_dt=0.01.h5')

#     plot_controls_and_strains(
# 'continuous_roll_mu_min=-3.0_mu_max=0.0_N_mu=10_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.01.h5')

#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         T_max = ('muscle', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         T_max = ('roll', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',        
#         T_max = ('muscle', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',        
#         T_max = ('roll', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_s0_t_min=0.1_s0_t_max=0.3_s0_t_step=0.01_A=5.0_f=5.0_N=100_dt=0.01.h5',
#         T_max = ('roll', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_s0_t_min=0.1_s0_t_max=0.3_s0_t_step=0.01_A=5.0_f=5.0_N=100_dt=0.01.h5',
#         T_max = ('muscle', 4))

#     plot_body_points_twist_angular_velocity(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         T_max = ('muscle', 4))

#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_mu_min=-3.0_mu_max=0.0_N_mu=10_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.01.h5',
#         T_max = ('roll', 4))
#
#     plot_roll_angle_twist_angular_velocity_chemogramm(
# 'continuous_roll_mu_min=-3.0_mu_max=0.0_N_mu=10_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.01.h5',
#         T_max = ('muscle', 4))

#     plot_body_points_twist_angular_velocity(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         T_max = ('muscle', 4))

#------------------------------------------------------------------------------ 
# Generate videos

    # N_worker = 1

#     generate_worm_videos(N_worker,
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5')

#     generate_worm_videos(N_worker,
# 'continuous_roll_mu_min=-3.0_mu_max=0.0_N_mu=10_A_=6.0_f=5.0_s0_t=0.25_s0_h=None_s0_t=0.25_N_100_dt_0.01.h5')

#------------------------------------------------------------------------------ 
# Animations
    
#     animate_head_trajectories(
# 'continuous_roll_A_min=1.0_A_max=7.0_A_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         'A_dv', T_max = 5.0, centre = True)
#
#     animate_head_trajectories(
# 'continuous_roll_f_min=1.0_f_max=8.0_f_step=1.0_N_100_dt_0.01_s0_h_None_s0_t_0.25.h5',
#         'f_dv', T_max = 5.0, centre = True)

    print('Finished!')
    






