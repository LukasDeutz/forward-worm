'''
Created on 14 Feb 2023

@author: lukas
'''
#Built-in
from argparse import ArgumentParser

# Third-party imports
import h5py
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.integrate import quad

# Local imports
from forward_worm.forward_undulation.undulations_dirs import log_dir, sim_dir, sweep_dir, fig_dir, video_dir
from parameter_scan import ParameterGrid
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_multiple_scalar_fields
from scipy.integrate._quadrature import trapezoid
from tensorboard.compat.proto import step_stats_pb2
import itertools

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

def plot_U_over_E_for_diffenrent_mu(h5_filename,
        show = False):
    
    h5, PG  = load_h5_file(h5_filename)
    E_arr = PG.v_from_key('E')
    mu_arr = PG.v_from_key('mu')
    
    U_mat = h5['U'][:]
    
    fig = plt.figure(figsize = (8, 12))
    gs = plt.GridSpec(2, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
                            
    for U_arr, mu in zip(U_mat, mu_arr):
               
        ax0.semilogx(E_arr, U_arr, label = fr'$\mu = {mu}$')
        ax1.semilogx(E_arr / mu, U_arr)
    
    ax0.legend()
    ax0.set_ylabel(r'$U$', fontsize = 20)
    ax0.set_xlabel(r'$E$', fontsize = 20)
    
    ax1.set_ylabel(r'$U$', fontsize = 20)
    ax1.set_xlabel(r'$\frac{E}{mu}$', fontsize = 20)
            
    if show: plt.show()
    
    return

def plot_U_and_W_over_lam_and_c(
        fig_path,
        lam_arr,
        c_arr,
        U,
        W,        
        show = False):
    '''
    Plots normalized undulation speed and mechanical muscle work done per 
    undulation cycle
    '''
                        
    # Fit maximal speed using spline
    # Use maximum speed on grid as initial guess 
    j_max, i_max = np.unravel_index(np.argmax(U), U.shape)
    x0 = np.array([lam_arr[i_max], c_arr[j_max]])                      
    
    func = RectBivariateSpline(lam_arr, c_arr, -U)    
    func_wrap = lambda x: func(x[0], x[1])[0]
    
    res = minimize(func_wrap, x0, 
        bounds=[(lam_arr.min(), lam_arr.max()), (c_arr.min(), c_arr.max())])
     
    # Plot    
    fig = plt.figure(figsize = (12, 6))
    gs = plt.GridSpec(3, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[1])
                        
    levels = 10        
    fz = 20 # axes label fontsize
    cb_fz = 16 # colorbar fontsize
                
    U, W = U.T, W.T 
            
    LAM, C  = np.meshgrid(lam_arr, c_arr)        
    CS = ax0.contourf(LAM, C, U, levels=levels, cmap = plt.cm.plasma)
    ax0.contour(LAM, C, U, levels, linestyles = '-', colors = ('k',))    
    ax0.plot(res.x[0], res.x[1], 'x', c = 'k', ms = '10')
    cbar = plt.colorbar(CS, ax = ax0)
    cbar.set_label('$U$', fontsize = cb_fz)
    
    CS = ax1.contourf(LAM, C, W, levels, cmap = mpl.cm.get_cmap('cividis'))
    ax1.contour(LAM, C, W, levels, linestyles = '-', colors = ('k',))    
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$W$', fontsize = cb_fz)

    CS = ax2.contourf(LAM, C, U / W, levels, cmap = mpl.cm.get_cmap('cividis'))
    ax2.contour(LAM, C, U / W, levels, linestyles = '-', colors = ('k',))    
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$W/U$', fontsize = cb_fz)

    ax0.set_ylabel(r'$c$', fontsize = fz)    
    ax1.set_ylabel(r'$c$', fontsize = fz)
    ax2.set_ylabel(r'$c$', fontsize = fz)    
    ax2.set_xlabel(r'$\lambda$', fontsize = fz)
                                                           
    if show: plt.show()
        
    plt.savefig(fig_path)

    return

def plot_U_and_W_for_different_x(
        h5_filename, 
        key,
        format_x = '{0:.2f}',
        get_x = None,
        show=False):
    '''
    Plot undulation speed U and work W_M done by the muscles per undulation 
    cycle over lambda and c for different x specified by given key    
    '''
    
    h5, PG  = load_h5_file(h5_filename)
    lam_arr = PG.v_from_key('lam')
    c_arr = PG.v_from_key('c')
    
    if get_x is None:
        x_arr = PG.v_from_key(key)
    else:
        x_arr = get_x(PG)

    U_arr = h5['U'][:]
    W_arr = h5['energies']['W'][:]

    dir = fig_dir / 'U_W' / Path(h5_filename)
    if not dir.exists(): dir.mkdir(parents = True)
     
    for x, U, W in zip(x_arr, U_arr, W_arr):
        
        fig_path = dir / f'{key}={format_x.format(x)}.png'        
       
        plot_U_and_W_over_lam_and_c(
            fig_path,
            lam_arr,
            c_arr,
            U,
            W,
            show=show)
    
    return
    
def plot_U_and_W_for_different_eta_nu(h5_filename, show=False):
    '''
    Plot undulation speed U and work W_M done by the muscles per undulation 
    cycle over lambda and c for different body viscosity eta and nu  
    '''
    def get_eta_over_E(PG):        
        eta_arr = PG.v_from_key('eta')
        E = PG.base_parameter['E']
        eta_over_E_arr = np.round(np.log10(eta_arr / E))
        return eta_over_E_arr
        
    plot_U_and_W_for_different_x(
        h5_filename, 
        key = 'eta_over_E', 
        x_format_str = '{x:.3f}', 
        get_x = get_eta_over_E, 
        show)
    
    return

#------------------------------------------------------------------------------ 
# Sainity checks
    
def plot_W_and_D_balance(
        fig_path,
        lam_arr,
        c_arr,
        W,
        D, 
        show = False):
        
    W, D = W.T, D.T
    LAM, C  = np.meshgrid(lam_arr, c_arr)        
                                                    
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
    CS = ax0.contourf(LAM, C, W, levels, cmap = plt.cm.plasma)
    ax0.contour(LAM, C, W, levels, colors = ('k',))                        
    ax0.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax0)
    cbar.set_label('$W$', fontsize = cb_fz)

    # Plot output energy        
    CS = ax1.contourf(LAM, C, -D, levels, cmap = plt.cm.plasma)
    ax1.contour(LAM, C, -D, levels, colors = ('k',))                        
    ax1.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax1)
    cbar.set_label('$-D$', fontsize = cb_fz)

    # Plot log of absolute relative error 
    err = np.log10(np.abs(D + W) / np.abs(W))

    CS = ax2.contourf(LAM, C, err, levels, cmap = plt.cm.cividis)
    ax2.contour(LAM, C, err, levels, linestyles = '-', colors = ('k',))    
    ax2.set_ylabel(r'$c$', fontsize = fz)        
    cbar = plt.colorbar(CS, ax = ax2)
    cbar.set_label('$\log(|W + D|/W)$', fontsize = cb_fz)
    ax2.set_xlabel(r'$\lambda$', fontsize = fz)        
                                                           
    if show: plt.show()
    
    plt.savefig(fig_path)    
    
    return
    
def plot_W_and_D_over_lam_c_for_differnet_x(
        h5_filename,
        key,
        format_x = '{x:.2f}',
        get_x = None,        
        show=False):

    h5, PG  = load_h5_file(h5_filename)
    lam_arr = PG.v_from_key('lam')
    c_arr = PG.v_from_key('c')
    
    if get_x is None:    
        x_arr = PG.v_from_key(key)
    else:
        x_arr = PG.get_x(PG)

    # first dimension iterates over f, second and 
    # third dimensions iterate over c and lam
    W_arr = h5['energies']['W'][:]
    D_arr = h5['energies']['D_I'][:] + h5['energies']['D_F'][:]

    dir = fig_dir / 'W_D_balance' / Path(h5_filename)
    if not dir.exists(): dir.mkdir(parents = True)
     
    for x, W, D in zip(x_arr, W_arr, D_arr):

        fig_path = dir / f'{key}={format_x.format(x)}.png'        
       
        plot_W_and_D_balance(
                fig_path,
                lam_arr,
                c_arr,
                W,
                D, 
                show = False)
        


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

    # h5_filename = ('undulation_'
    #     'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
    #     'c_min=0.5_c_max=1.6_c_step=0.1_'
    #     'f=2.0_mu_0.001_N=250_dt=0.001.h5')
    
    # h5_filename = (f'undulation_'
    #     'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
    #     'c_min=0.5_c_max=1.6_c_step=0.1'
    #     '_f=2.0_mu_0.001_N=250_dt=0.0001.h5')
    
    h5_filename_E_mu = ('speed_and_energies_'
        'mu_min=-3_mu_max=1_N_mu=5_'
        'E_min=1_E_max=7_N_E=30_'
        'A_4.0_lam_1.0_f=2.0_N=100_dt=0.01.h5')    
    

    h5_filename_E_lam_c = ('speed_and_energies_'
        'E_min=2_E_max=5_E_step=1_'
        'c_min=0.5_c_max=1.6_c_step=0.1_'
        'f=2.0_mu_0.001_N=250_dt=0.001.h5')

    h5_filename_eta_nu = ('undulation_'
        'eta_min=-3_eta_max=0.0_eta_step=1.0_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'c_min=0.5_c_max=2.0_c_step=0.1_'
        'f=2.0_mu_0.001_N=250_dt=0.001.h5')

    h5_filename_f = ('speed_and_energies_'
        'f_min=0.5_f_max=2.0_f_step=0.5_'
        'lam_min=0.5_lam_max=2.0_lam_step=0.1_'
        'c_min=0.5_c_max=2.0_c_step=0.1_'
        'mu_0.001_N=250_dt=0.001.h5')
    
    #plot_U_and_W_for_different_f(h5_filename_f)
    # plot_W_and_D_over_lam_c_for_differnet_f(h5_filename_f)    
    plot_W_and_D_over_lam_c_for_differnet_x(h5_filename_E_lam_c, 'E', format_x = '{0:.0f}')            
    # plot_W_over_U_for_different_f(h5_filename_f)
    #plot_U_over_E_for_diffenrent_mu(h5_filename_E_mu, show = True)
        
    #plot_U_and_W_for_different_eta_nu(h5_filename_eta_nu, show=False)
    
    #plot_W_M_E_over_f(h5_filename_f)    
        
    #plot_W_M_E_over_lam_c_for_differnet_eta_nu(
    #     h5_filename_eta_nu)    
    
    #plot_undulation_speed_and_muscle_work(h5_filename, show = True)            
    # plot_input_output_energy_balance(h5_filename, show = True)
    
    # plot_true_powers(h5_filename, show=True)
    
    #plot_energy_costs(h5_filenames, c_arr, show = False)
        
    #plot_power_balance(h5_filenames[5])

    # plot_true_powers(h5_filenames[5], show = True)
        
    print('Finished!')
    
    
#------------------------------------------------------------------------------ 
# Old code

# def plot_energy_costs(h5_filenames, c_arr, show = False):
#
#     '''
#     Plots normalized undulation speed and mechanical muscle work done per 
#     undulation cycle
#     '''
#
#     h5, PG  = load_h5_file(h5_filenames[0])
#     lam_arr = PG.v_from_key('lam')
#     LAM, C  = np.meshgrid(lam_arr, c_arr)        
#
#     E_out = np.zeros_like(LAM)  
#     V = np.zeros_like(LAM)
#     D = np.zeros_like(LAM)
#     W_F = np.zeros_like(LAM)
#
#     for i, fn in enumerate(h5_filenames):
#
#         h5, PG = load_h5_file(fn)
#
#         E_out[i, :] = h5['energy-costs']['E_out'][:]
#         V[i, :] = h5['energy-costs']['V'][:]
#         D[i, :] = h5['energy-costs']['D'][:]
#         W_F[i, :] = h5['energy-costs']['W_F'][:]
#
#     # Plot    
#     fig, axs = plt.subplots(2, 3, figsize=(18, 12))
#
#     levels = 10        
#     fz = 20 # axes label fontsize
#     cb_fz = 16 # colorbar fontsize
#
#     norm = mpl.colors.Normalize(vmin=0, vmax=1)
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)  
#
#     labels = ['$V$', '$D$', '$W_\mathrm{F}$']
#
#
#     for i, E in enumerate([V, D, W_F]):
#
#         ax0i = axs[0, i]
#         ax1i = axs[1, i]
#
#         CS = ax0i.contourf(LAM, C, E, levels, cmap = plt.cm.plasma)
#         ax0i.contour(LAM, C, E, levels, linestyles = '-', colors = ('k',))    
#         cbar = plt.colorbar(CS, ax = ax0i)
#         cbar.set_label(labels[i], fontsize = cb_fz)
#
#         CS = ax1i.contourf(LAM, C, np.abs(E/E_out), levels, cmap = plt.cm.Reds, norm = norm)
#         ax1i.contour(LAM, C, np.abs(E/E_out), levels, linestyles = '-', colors = ('k',))    
#         ax1i.set_xlabel('$\lambda$', fontsize = fz)
#
#         cbar = plt.colorbar(sm, ax = ax1i)
#         cbar.remove()          
#
#     axs[0, 0].set_ylabel('$c$', fontsize = fz)
#     axs[1, 0].set_ylabel('$c$', fontsize = fz)
#
#     cbar = fig.colorbar(sm, ax = axs[1,:], orientation = 'horizontal')                        
#     cbar.set_label('$E/E_\mathrm{out}$', fontsize = cb_fz)
#
#     if show: plt.show()
#
#     filename =('energy_costs_lam_c_'
#         f'N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
#     plt.savefig(fig_dir / filename)
#
#     return
# def plot_W_M_E_balance(
#         fig_path,
#         lam_arr,
#         c_arr,
#         W_M_mat,
#         E_out_mat, 
#         show = False):
#     '''
#     Plots work done by the muscles per undulation cycle, energy lost 
#     per undulation cycle and relative error between the two    
#     '''
#
#     W_M_mat, E_out_mat = W_M_mat.T, E_out_mat.T
#     LAM, C  = np.meshgrid(lam_arr, c_arr)        
#
#     # Plot    
#     fig = plt.figure(figsize = (10, 14))
#     gs = plt.GridSpec(3, 1)
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[2])
#
#     fz = 20 # axes label fontsize
#     cb_fz = 18 # colobar labe fontsize
#
#     levels = 10        
#     # Plot mechanical work one per undulation cycle        
#     CS = ax0.contourf(LAM, C, W_M_mat, levels, cmap = plt.cm.plasma)
#     ax0.contour(LAM, C, W_M_mat, levels, colors = ('k',))                        
#     ax0.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax0)
#     cbar.set_label('$W_\mathrm{M}$', fontsize = cb_fz)
#
#     # Plot output energy        
#     CS = ax1.contourf(LAM, C, E_out_mat, levels, cmap = plt.cm.plasma)
#     ax1.contour(LAM, C, E_out_mat, levels, colors = ('k',))                        
#     ax1.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax1)
#     cbar.set_label('$E_\mathrm{out}$', fontsize = cb_fz)
#
#     # Plot log of absolute relative error 
#     err = np.log10(np.abs(E_out_mat - W_M_mat) / np.abs(W_M_mat))
#
#     CS = ax2.contourf(LAM, C, err, levels, cmap = plt.cm.cividis)
#     ax2.contour(LAM, C, err, levels, linestyles = '-', colors = ('k',))    
#     ax2.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax2)
#     cbar.set_label('$\log(\mathrm{rel Error})$', fontsize = cb_fz)
#     ax2.set_xlabel(r'$\lambda$', fontsize = fz)        
#
#     if show: plt.show()
#
#     plt.savefig(fig_path)
#
#     return    
#
# def plot_W_M_E_over_lam_c_for_differnet_f(
#         h5_filename,
#         show=False):
#
#     h5, PG  = load_h5_file(h5_filename)
#     lam_arr = PG.v_from_key('lam')
#     c_arr = PG.v_from_key('c')
#     f_arr = PG.v_from_key('f')
#
#     # rows iterate over lambda, columns over c
#     W_M = h5['energies']['abs_W_M'][:].reshape(PG.shape)
#     E_out = h5['energies']['E_out'][:].reshape(PG.shape)
#
#     dir = fig_dir / 'W_M_E_balance' / Path(h5_filename)
#     if not dir.exists(): dir.mkdir(parents = True)
#
#     for f, W_M_mat, E_out_mat in zip(f_arr, W_M, E_out):
#
#        fig_path = dir / f'f={f:.2f}.png'        
#
#        plot_W_M_E_balance(
#             fig_path,
#             lam_arr,
#             c_arr,
#             W_M_mat,
#             E_out_mat,            
#             show=show)
#
# def plot_W_M_E_over_lam_c_for_differnet_eta_nu(
#         h5_filename,
#         show=False):
#
#     h5, PG  = load_h5_file(h5_filename)
#     lam_arr = PG.v_from_key('lam')
#     c_arr = PG.v_from_key('c')
#     eta_arr = PG.v_from_key('eta')
#     E = PG.base_parameter['E']
#     eta_over_E_arr = np.round(np.log10(eta_arr / E))
#
#     # rows iterate over lambda, columns over c
#     W_M = h5['energies']['abs_W_M'][:].reshape(PG.shape)
#     E_out = h5['energies']['E_out'][:].reshape(PG.shape)
#
#     dir = fig_dir / 'W_M_E_balance' / Path(h5_filename)
#     if not dir.exists(): dir.mkdir(parents = True)
#
#     for eta_over_E, W_M_mat, E_out_mat in zip(eta_over_E_arr, W_M, E_out):
#
#        fig_path = dir / f'f={eta_over_E:.1f}.png'        
#
#        plot_W_M_E_balance(
#             fig_path,
#             lam_arr,
#             c_arr,
#             W_M_mat,
#             E_out_mat,            
#             show=show)
#
# def plot_true_energy_costs(h5_filename, c_arr, show = False):
#     '''
#     True energy costs are computed by distinguishing between potential energy 
#     that is stored or released at any point in time.
#
#     If the released potential is larger than the dissipation rate then
#     the muscles need to work against the passive restoring forces and 
#     we assign the energy cost to the change in potential energy
#
#     If the released potential energy is smaller than the disspation rate
#     then we substract the energy from the dissipation rate which gives as
#     the true dissipation rate which needs to be balanced by the muscle work           
#
#     The summed true energy costs need to balance the muscle work
#
#     :param h5_filenames (List[str]): List with hdf5 filenames
#     :param c_arr (np.array): A/q array
#     :param show (bool): If true, show plot
#     '''
#
#     h5, PG  = load_h5_file(h5_filename)
#     lam_arr = PG.v_from_key('lam')
#     c_arr = PG.v_from_key('c')
#     LAM, C  = np.meshgrid(lam_arr, c_arr)        
#
#     W_M_mat = np.zeros_like(LAM)
#     E_out_mat = np.zeros_like(LAM)
#
#     # rows iterate over lambda, columns over c
#     W_M_mat = h5['energies']['abs_W_M'][:].reshape(PG.shape).T
#     E_out_mat = h5['energies']['E_out'][:].reshape(PG.shape).T
#
#     # Plot    
#     fig = plt.figure(figsize = (10, 14))
#     gs = plt.GridSpec(3, 1)
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[2])
#
#     fz = 20 # axes label fontsize
#     cb_fz = 18 # colobar labe fontsize
#
#     levels = 10        
#     # Plot mechanical work one per undulation cycle        
#     CS = ax0.contourf(LAM, C, W_M_mat, levels, cmap = plt.cm.plasma)
#     ax0.contour(LAM, C, W_M_mat, levels, colors = ('k',))                        
#     ax0.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax0)
#     cbar.set_label('$W_\mathrm{M}$', fontsize = cb_fz)
#
#     # Plot output energy        
#     CS = ax1.contourf(LAM, C, E_out_mat, levels, cmap = plt.cm.plasma)
#     ax1.contour(LAM, C, E_out_mat, levels, colors = ('k',))                        
#     ax1.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax1)
#     cbar.set_label('$E_\mathrm{out}$', fontsize = cb_fz)
#
#     # Plot log of absolute relative error 
#     err = np.log10(np.abs(E_out_mat - W_M_mat) / np.abs(W_M_mat))
#
#     CS = ax2.contourf(LAM, C, err, levels, cmap = plt.cm.cividis)
#     ax2.contour(LAM, C, err, levels, linestyles = '-', colors = ('k',))    
#     ax1.set_ylabel(r'$c$', fontsize = fz)        
#     cbar = plt.colorbar(CS, ax = ax2)
#     cbar.set_label('$\log(\mathrm{rel Error})$', fontsize = cb_fz)
#
#     if show: plt.show()
#
#     filename = (f'input_vs_true_ouput_energy_balance_lam_c'
#         f'_N={PG.base_parameter["N"]}_dt={PG.base_parameter["dt"]}.png')
#     plt.savefig(fig_dir / filename)
#
#     return        
#
# def plot_power_balance(h5_filename, 
#         N_plot = 4,
#         show = False):
#
#     h5, PG  = load_h5_file(h5_filenames[0])
#     lam_arr = PG.v_from_key('lam')    
#     T_undu = 1.0 / PG.base_parameter['f']
#
#     N_sim = h5['FS']['V_dot_k'].shape[0]
#
#     if N_plot > N_sim: N_plot = N_sim
#
#     sim_idx_arr = np.linspace(0, N_sim-1, N_plot, dtype = int)
#
#     plt.figure(figsize = (18, N_plot*6))
#     gs = plt.GridSpec(N_plot, 3)
#
#     powers, t = EPP.powers_from_h5(h5, t_start = T_undu)
#
#     for i, sim_idx in enumerate(sim_idx_arr):
#
#         ax0 = plt.subplot(gs[i, 0])
#         ax1 = plt.subplot(gs[i, 1])
#         ax2 = plt.subplot(gs[i, 2])
#
#         dot_V = powers['dot_V_k'][sim_idx, :] + powers['dot_V_sig'][sim_idx, :]
#         dot_D = powers['dot_D_k'][sim_idx, :] + powers['dot_D_sig'][sim_idx, :]                                        
#         dot_W_F = powers['dot_W_F_F'][sim_idx, :] + powers['dot_W_F_T'][sim_idx, :]                                        
#         dot_W_M = powers['dot_W_M_F'][sim_idx, :] + powers['dot_W_M_T'][sim_idx, :]
#
#         ax0.plot(t, dot_V, label = r'$\dot{V}$', c = 'g')
#         ax0.plot(t, dot_D, label = r'$\dot{D}$', c = 'r')
#         ax0.plot(t, dot_W_F, label = r'$\dot{W}_F$', c = 'b')
#         ax0.plot(t, dot_W_M, label = r'$\dot{W}_M$', c = 'orange')
#
#         ax1.plot(t, -dot_V + dot_D + dot_W_F, c = 'k')
#         ax1.plot(t, dot_W_M, c = 'g')
#         ax2.semilogy(t, np.abs(-dot_V + dot_D + dot_W_F + dot_W_M) / np.abs(dot_W_M), c = 'k')
#
#         ax0.set_ylabel(fr'$\lambda = {lam_arr[sim_idx]:.2f}$', fontsize = 20)
#
#     ax0.set_xlabel(r'$t$', fontsize = 20)
#     ax1.set_xlabel(r'$t$', fontsize = 20)
#     ax2.set_xlabel(r'$t$', fontsize = 20)
#
#     ax0.legend()                
#     plt.show()
#
#     filename = 'input_output_power_balance_lam_c.png'
#     plt.savefig(fig_dir / filename)
#
#     return
#
# def plot_true_powers(
#         h5_filename, 
#         N_plot = 4,
#         show = False
#     ):
#
#     h5, PG  = load_h5_file(h5_filename)
#     lam_arr = PG.v_from_key('lam')    
#     c_arr = PG.v_from_key('c')    
#     LAM, C  = np.meshgrid(lam_arr, c_arr)        
#
#     W_M_mat = h5['energies']['D'][:].reshape(PG.shape).T
#
#     lam_idx_arr = np.linspace(0, PG.shape[0], N_plot, 
#         endpoint=False, dtype = int)
#     c_idx_arr = np.linspace(0, PG.shape[1], N_plot,
#         endpoint=False, dtype = int)
#
#     T_undu = 1.0 / PG.base_parameter['f']
#
#     N_undu = np.round(PG.base_parameter['T'] / T_undu - 1.0)
#
#     plt.figure(figsize = (N_plot*6, 2*N_plot*6))
#     gs = plt.GridSpec(N_plot, 2*N_plot)    
#     ax0 = plt.subplot(gs[:N_plot, :N_plot])
#
#     levels = 10
#     LAM, C  = np.meshgrid(lam_arr, c_arr)        
#     CS = ax0.contourf(LAM, C, W_M_mat, levels, cmap = plt.cm.plasma)
#     ax0.contour(LAM, C, W_M_mat, levels, linestyles = '-', colors = ('k',))    
#     cbar = plt.colorbar(CS, ax = ax0)
#     cbar.set_label(r'$|W|$', fontsize = 18)
#
#     for lam in lam_arr[lam_idx_arr]: 
#         for c in c_arr[c_idx_arr]:
#             ax0.plot(lam, c, 'x', c='k', ms=10)
#
#     powers, t = EPP.powers_from_h5(h5, t_start = 2*T_undu)
#     dot_V_arr, dot_D_arr, dot_W_F_arr = EPP.comp_true_powers(powers)                 
#
#     #dot_W_M_arr = np.abs(np.abs(powers['dot_W_M_F'] + powers['dot_W_M_T']))
#     for i, lam_idx in enumerate(lam_idx_arr):
#         for j, c_idx in enumerate(c_idx_arr):
#
#             k = np.ravel_multi_index((i, j), PG.shape)
#             ax = plt.subplot(gs[N_plot-j-1, N_plot+i])            
#             ax.plot(t, dot_V_arr[k, :], c ='g')
#             ax.plot(t, dot_D_arr[k, :], c = 'r')
#             #ax.plot(t, dot_W_F_arr[k, :], c = 'b')
#
#     if show: plt.show()                                                    
#     filename = 'input_output_power_balance_lam_c.png'
#     plt.savefig(fig_dir / filename)
#
#     return




