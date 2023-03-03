'''
Created on 14 Feb 2023

@author: lukas
'''
# Third-party imports
import h5py
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

# Local imports
from forward_worm.forward_undulation.undulations_dirs import log_dir, sim_dir, sweep_dir, fig_dir, video_dir
from parameter_scan import ParameterGrid
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_multiple_scalar_fields
from scipy.integrate._quadrature import trapezoid

#------------------------------------------------------------------------------ 

def load_h5_file(filename):

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

#------------------------------------------------------------------------------ 

def generate_worm_videos(h5_filename):

    from simple_worm_experiments.worm_studio import WormStudio

    h5_filepath = sweep_dir / h5_filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    vid_filenames = [f'lam={p["lam"]}_A={p["A"]}_f_{p["f"]}_'
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
# Results

def plot_undulation_speed_and_muscle_work(h5_filename, show = False):
    '''
    Plots normalized undulation speed and mechanical muscle work done per 
    undulation cycle
    '''
            
    h5, PG  = load_h5_file(h5_filename)
    lam_arr = PG.v_from_key('lam')
    c_arr = PG.v_from_key('c')
        
    # rows iterate over lambda, columns over c
    U_mat = h5['U'][:].reshape(PG.shape)
    W_M_mat = h5['energies']['abs_W_M'][:].reshape(PG.shape)
            
    # Fit maximal speed using spline
    # Use maximum speed on grid as initial guess 
    j_max, i_max = np.unravel_index(np.argmax(U_mat), U_mat.shape)
    x0 = np.array([lam_arr[i_max], c_arr[j_max]])                      
    
    func = RectBivariateSpline(lam_arr, c_arr, -U_mat)    
    func_wrap = lambda x: func(x[0], x[1])[0]
    
    res = minimize(func_wrap, x0)
                    
    # Plot    
    fig = plt.figure(figsize = (12, 6))
    gs = plt.GridSpec(1, 2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
                        
    levels = 10        
    fz = 20 # axes label fontsize
    cb_fz = 16 # colorbar fontsize
    
    LAM, C  = np.meshgrid(lam_arr, c_arr)        
    CS = ax0.contourf(LAM, C, U_mat.T, levels, cmap = plt.cm.plasma)
    ax0.contour(LAM, C, U_mat.T, levels, linestyles = '-', colors = ('k',))    
    ax0.plot(res.x[0], res.x[1], 'x', c = 'k', ms = '10')
    cbar = plt.colorbar(CS, ax = ax0)
    cbar.set_label('$U$', fontsize = cb_fz)
    
    CS = ax1.contourf(LAM, C, W_M_mat.T, levels, cmap = mpl.cm.get_cmap('cividis'))
    ax1.contour(LAM, C, W_M_mat.T, levels, linestyles = '-', colors = ('k',))    
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$W_\mathrm{M}$', fontsize = cb_fz)
                
    ax0.set_xlabel(r'$\lambda$', fontsize = fz)
    ax0.set_ylabel(r'$c$', fontsize = fz)    
    ax1.set_xlabel(r'$\lambda$', fontsize = fz)
                                                           
    if show: plt.show()
    
    filename =('swimming_speed_and_muscle_work_lam_c_'
        f'N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
    plt.savefig(fig_dir / filename)

    return

def plot_energy_costs(h5_filenames, c_arr, show = False):

    '''
    Plots normalized undulation speed and mechanical muscle work done per 
    undulation cycle
    '''
            
    h5, PG  = load_h5_file(h5_filenames[0])
    lam_arr = PG.v_from_key('lam')
    LAM, C  = np.meshgrid(lam_arr, c_arr)        
    
    E_out = np.zeros_like(LAM)  
    V = np.zeros_like(LAM)
    D = np.zeros_like(LAM)
    W_F = np.zeros_like(LAM)
    
    for i, fn in enumerate(h5_filenames):
    
        h5, PG = load_h5_file(fn)
        
        E_out[i, :] = h5['energy-costs']['E_out'][:]
        V[i, :] = h5['energy-costs']['V'][:]
        D[i, :] = h5['energy-costs']['D'][:]
        W_F[i, :] = h5['energy-costs']['W_F'][:]

    # Plot    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    levels = 10        
    fz = 20 # axes label fontsize
    cb_fz = 16 # colorbar fontsize
    
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)  
            
    labels = ['$V$', '$D$', '$W_\mathrm{F}$']
    
    
    for i, E in enumerate([V, D, W_F]):
        
        ax0i = axs[0, i]
        ax1i = axs[1, i]
                
        CS = ax0i.contourf(LAM, C, E, levels, cmap = plt.cm.plasma)
        ax0i.contour(LAM, C, E, levels, linestyles = '-', colors = ('k',))    
        cbar = plt.colorbar(CS, ax = ax0i)
        cbar.set_label(labels[i], fontsize = cb_fz)
    
        CS = ax1i.contourf(LAM, C, np.abs(E/E_out), levels, cmap = plt.cm.Reds, norm = norm)
        ax1i.contour(LAM, C, np.abs(E/E_out), levels, linestyles = '-', colors = ('k',))    
        ax1i.set_xlabel('$\lambda$', fontsize = fz)
        
        cbar = plt.colorbar(sm, ax = ax1i)
        cbar.remove()          
        
    axs[0, 0].set_ylabel('$c$', fontsize = fz)
    axs[1, 0].set_ylabel('$c$', fontsize = fz)

    cbar = fig.colorbar(sm, ax = axs[1,:], orientation = 'horizontal')                        
    cbar.set_label('$E/E_\mathrm{out}$', fontsize = cb_fz)
                             
    if show: plt.show()
    
    filename =('energy_costs_lam_c_'
        f'N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
    plt.savefig(fig_dir / filename)

    return


#------------------------------------------------------------------------------ 
# Sainity checks
    
def plot_input_output_energy_balance(h5_filename, show = False):
    '''
    Plots work done by the muscles per undulation cycle, energy lost 
    per undulation cycle and relative error between the two
    
    :param h5_filenames (List[str]): List with hdf5 filenames
    :param c_arr (np.array): A/q array
    :param show (bool): If true, show plot
    '''
        
    h5, PG  = load_h5_file(h5_filename)
    lam_arr = PG.v_from_key('lam')
    c_arr = PG.v_from_key('c')
    LAM, C  = np.meshgrid(lam_arr, c_arr)        

    # rows iterate over lambda, columns over c
    W_M_mat = h5['energies']['abs_W_M'][:].reshape(PG.shape).T
    E_out_mat = h5['energies']['E_out'][:].reshape(PG.shape).T
                                                    
    # Plot    
    fig = plt.figure(figsize = (10, 14))
    gs = plt.GridSpec(3, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    fz = 20 # axes label fontsize
    cb_fz = 18 # colobar labe fontsize
                                                      
    levels = 10        
    # Plot mechanical work one per undulation cycle        
    CS = ax0.contourf(LAM, C, W_M_mat, levels, cmap = plt.cm.plasma)
    ax0.contour(LAM, C, W_M_mat, levels, colors = ('k',))                        
    ax0.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax0)
    cbar.set_label('$W_\mathrm{M}$', fontsize = cb_fz)

    # Plot output energy        
    CS = ax1.contourf(LAM, C, E_out_mat, levels, cmap = plt.cm.plasma)
    ax1.contour(LAM, C, E_out_mat, levels, colors = ('k',))                        
    ax1.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$E_\mathrm{out}$', fontsize = cb_fz)

    # Plot log of absolute relative error 
    err = np.log10(np.abs(E_out_mat - W_M_mat) / np.abs(W_M_mat))

    CS = ax2.contourf(LAM, C, err, levels, cmap = plt.cm.cividis)
    ax2.contour(LAM, C, err, levels, linestyles = '-', colors = ('k',))    
    ax1.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax2)
    cbar.set_label('$\log(\mathrm{rel Error})$', fontsize = cb_fz)
                                                           
    if show: plt.show()
    
    filename = (f'input_output_energy_balance_lam_c'
        f'_N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
    plt.savefig(fig_dir / filename)

    return    

def plot_true_energy_costs(h5_filenames, c_arr, show = False):
    '''
    True energy costs are computed by accounting for released potential
    at any point in time.
    
    If the released potential is larger than the dissipation rate then
    the muscles need to work against the passive restoring forces and 
    we assign the energy cost to the change in potential energy
    
    If the released potential energy is smaller than the disspation rate
    then we substract the energy from the dissipation rate which gives as
    the true dissipation rate which needs to be balanced by the muscle work           
        
    The summed true energy costs need to balance the muscle work
        
    :param h5_filenames (List[str]): List with hdf5 filenames
    :param c_arr (np.array): A/q array
    :param show (bool): If true, show plot
    '''
        
    h5, PG  = load_h5_file(h5_filenames[0])
    lam_arr = PG.v_from_key('lam')
    LAM, C  = np.meshgrid(lam_arr, c_arr)        

    T_undu = 1.0 / PG.base_parameter['f']    

    W_M_mat = np.zeros_like(LAM)
    E_out_mat = np.zeros_like(LAM)
                                              
    for i, fn in enumerate(h5_filenames):
    
        h5, _ = load_h5_file(fn)
                
        W_M_mat[i, :] = h5['energies']['abs_W_M'][:]           
        E_out_mat[i, :] = -h5['energy-costs']['E_out'][:]
                                                            
    # Plot    
    fig = plt.figure(figsize = (10, 14))
    gs = plt.GridSpec(3, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    fz = 20 # axes label fontsize
    cb_fz = 18 # colobar labe fontsize
                                                      
    levels = 10        
    # Plot mechanical work one per undulation cycle        
    CS = ax0.contourf(LAM, C, W_M_mat, levels, cmap = plt.cm.plasma)
    ax0.contour(LAM, C, W_M_mat, levels, colors = ('k',))                        
    ax0.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax0)
    cbar.set_label('$W_\mathrm{M}$', fontsize = cb_fz)

    # Plot output energy        
    CS = ax1.contourf(LAM, C, E_out_mat, levels, cmap = plt.cm.plasma)
    ax1.contour(LAM, C, E_out_mat, levels, colors = ('k',))                        
    ax1.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$E_\mathrm{out}$', fontsize = cb_fz)

    # Plot log of absolute relative error 
    err = np.log10(np.abs(E_out_mat - W_M_mat) / np.abs(W_M_mat))

    CS = ax2.contourf(LAM, C, err, levels, cmap = plt.cm.cividis)
    ax2.contour(LAM, C, err, levels, linestyles = '-', colors = ('k',))    
    ax1.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax2)
    cbar.set_label('$\log(\mathrm{rel Error})$', fontsize = cb_fz)
                                                           
    if show: plt.show()
    
    filename = (f'input_vs_true_ouput_energy_balance_lam_c'
        f'_N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
    plt.savefig(fig_dir / filename)

    return        

def plot_power_balance(h5_filename, 
        N_plot = 4,
        show = False):
    
    h5, PG  = load_h5_file(h5_filenames[0])
    lam_arr = PG.v_from_key('lam')    
    T_undu = 1.0 / PG.base_parameter['f']
        
    N_sim = h5['FS']['V_dot_k'].shape[0]
    
    if N_plot > N_sim: N_plot = N_sim
    
    sim_idx_arr = np.linspace(0, N_sim-1, N_plot, dtype = int)

    plt.figure(figsize = (18, N_plot*6))
    gs = plt.GridSpec(N_plot, 3)

    powers, t = EPP.powers_from_h5(h5, t_start = T_undu)
            
    for i, sim_idx in enumerate(sim_idx_arr):
                
        ax0 = plt.subplot(gs[i, 0])
        ax1 = plt.subplot(gs[i, 1])
        ax2 = plt.subplot(gs[i, 2])
                
        dot_V = powers['dot_V_k'][sim_idx, :] + powers['dot_V_sig'][sim_idx, :]
        dot_D = powers['dot_D_k'][sim_idx, :] + powers['dot_D_sig'][sim_idx, :]                                        
        dot_W_F = powers['dot_W_F_F'][sim_idx, :] + powers['dot_W_F_T'][sim_idx, :]                                        
        dot_W_M = powers['dot_W_M_F'][sim_idx, :] + powers['dot_W_M_T'][sim_idx, :]
                        
        ax0.plot(t, dot_V, label = r'$\dot{V}$', c = 'g')
        ax0.plot(t, dot_D, label = r'$\dot{D}$', c = 'r')
        ax0.plot(t, dot_W_F, label = r'$\dot{W}_F$', c = 'b')
        ax0.plot(t, dot_W_M, label = r'$\dot{W}_M$', c = 'orange')

        ax1.plot(t, -dot_V + dot_D + dot_W_F, c = 'k')
        ax1.plot(t, dot_W_M, c = 'g')
        ax2.semilogy(t, np.abs(-dot_V + dot_D + dot_W_F + dot_W_M) / np.abs(dot_W_M), c = 'k')

        ax0.set_ylabel(fr'$\lambda = {lam_arr[sim_idx]:.2f}$', fontsize = 20)

    ax0.set_xlabel(r'$t$', fontsize = 20)
    ax1.set_xlabel(r'$t$', fontsize = 20)
    ax2.set_xlabel(r'$t$', fontsize = 20)
    
    ax0.legend()                
    plt.show()

    filename = 'input_output_power_balance_lam_c.png'
    plt.savefig(fig_dir / filename)
            
    return

def plot_true_powers(h5_filename, 
        N_plot = 4,
        show = False):
    
    h5, PG  = load_h5_file(h5_filenames[0])
    lam_arr = PG.v_from_key('lam')    
    T_undu = 1.0 / PG.base_parameter['f']
        
    N_undu = np.round(PG.base_parameter['T'] / T_undu - 1.0)
                
    N_sim = h5['FS']['V_dot_k'].shape[0]
    
    if N_plot > N_sim: N_plot = N_sim
    
    sim_idx_arr = np.linspace(0, N_sim-1, N_plot, dtype = int)

    plt.figure(figsize = (12, N_plot*6))
    gs = plt.GridSpec(N_plot, 2)

    powers, t = EPP.powers_from_h5(h5, t_start = T_undu)
    dt = t[1]-t[0]
            
    dot_V_arr, dot_D_arr, dot_W_F_arr = EPP.comp_true_powers(powers)     
    dot_W_M_arr = - np.abs(powers['dot_W_M_F'] + powers['dot_W_M_T'])
        
    for i, sim_idx in enumerate(sim_idx_arr):
                
        ax0 = plt.subplot(gs[i, 0])
        ax1 = plt.subplot(gs[i, 1])
        # ax2 = plt.subplot(gs[i, 2])
                         
        dot_V = dot_V_arr[sim_idx, :] 
        dot_D = dot_D_arr[sim_idx, :]
        dot_W_F = dot_W_F_arr[sim_idx, :]                                         
        ax0.plot(t, dot_V, label = r'$\dot{V}$', c = 'g')
        ax0.plot(t, dot_D, label = r'$\dot{D}$', c = 'r')
        ax0.plot(t, dot_W_F, label = r'$\dot{W}_F$', c = 'b')
        
        ax1.plot(t, dot_W_M_arr[sim_idx, :])
        ax1.plot(t, dot_V + dot_D + dot_W_F)
                                
        # V = trapezoid(dot_V, dx = dt) / N_undu
        # D = trapezoid(dot_V, dx = dt) / N_undu
        # W_F = trapezoid(dot_W_F, dx = dt) / N_undu

        # ax1.plot(t, -dot_V + dot_D + dot_W_F, c = 'k')
        # ax1.plot(t, dot_W_M, c = 'g')
        # ax2.semilogy(t, np.abs(-dot_V + dot_D + dot_W_F + dot_W_M) / np.abs(dot_W_M), c = 'k')

        ax0.set_ylabel(fr'$\lambda = {lam_arr[sim_idx]:.2f}$', fontsize = 20)

    ax0.set_xlabel(r'$t$', fontsize = 20)
    # ax1.set_xlabel(r'$t$', fontsize = 20)
    # ax2.set_xlabel(r'$t$', fontsize = 20)
    
    ax0.legend()                
    plt.show()

    filename = 'input_output_power_balance_lam_c.png'
    plt.savefig(fig_dir / filename)

           



if __name__ == '__main__':
        
#     generate_worm_videos(
# 'undulation_rft_A_min=1.0_A_max=8.01_A_step=1.0_f=2.0_N=100_dt=0.01.h5')

#     generate_worm_videos(
# f'undulation_lam_min=0.5_lam_max=2.0_lam_step=0.1_c=1.40_f=2.0_mu_0.001_N=100_dt=0.01.h5') 

#------------------------------------------------------------------------------ 
# Plotting    
    
    # h5_filename = (f'undulation_'
    #     f'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
    #     f'c_min=0.5_c_max=1.6_c_step=0.1_f=2.0_'
    #     f'mu_0.001_N=100_dt=0.01.h5'
    #     )
    
    h5_filename = (f'undulation_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'c_min=0.5_c_max=1.6_c_step=0.1'
        '_f=2.0_mu_0.001_N=250_dt=0.0001.h5')

    plot_undulation_speed_and_muscle_work(h5_filename, show = True)        
    
    plot_input_output_energy_balance(h5_filenames, c_arr, show = False)
    #plot_energy_costs(h5_filenames, c_arr, show = False)
        
    #plot_power_balance(h5_filenames[5])

    # plot_true_powers(h5_filenames[5], show = True)
    
        
    print('Finished!')
    
    
    




