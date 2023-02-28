

'''
Created on 6 Dec 2022

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
from forward_worm.contraction_relaxation.contraction_relaxation_dirs import log_dir, sim_dir, sweep_dir, fig_dir, video_dir
from parameter_scan import ParameterGrid
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_multiple_scalar_fields

#------------------------------------------------------------------------------ 

def load_h5_file(filename):

    h5_filepath = sweep_dir / filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    return h5, PG

def plot_energy_conservation_for_different_dt_and_N(
        h5_filepath_dt,
        h5_filepath_N, 
        show = False):

    h5_dt, PG_dt = load_h5_file(h5_filepath_dt)
    h5_N, PG_N = load_h5_file(h5_filepath_N)

    dt_arr = PG_dt.v_from_key('dt')
    N_arr = PG_N.v_from_key('N')
        
    plt.figure(figsize = (16, 12))                
    gs = plt.GridSpec(2, 2)
    
    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    ax01 = plt.subplot(gs[0,1])
    ax11 = plt.subplot(gs[1,1])
                
    for i, (h5, v_arr, label) in enumerate(zip([h5_dt, h5_N], [dt_arr, N_arr], 
            [r'$dt$', r'$N$'])):
                                                    
        t = h5['t'][:]
        E_out_arr = h5['powers']['dot_E_out'][:]
        dot_W_M_arr = h5['powers']['dot_W_M'][:]

        cm = plt.cm.plasma
        norm = mpl.colors.LogNorm(vmin=np.min(v_arr), vmax=np.max(v_arr))    
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)    
        colors = sm.to_rgba(v_arr)
        
        ax0i = plt.subplot(gs[0, i])
        
        # Plot powers for highest resolution 
        ax0i.plot(t, E_out_arr[-1, :], label = r'$\dot{E}_\mathrm{out}$')
        ax0i.plot(t, dot_W_M_arr[-1, :], label = r'$\dot{W}_\mathrm{M}$')        
        ax0i.legend(fontsize = 16)
    
        ax1i = plt.subplot(gs[1,i])
    
        # Threshold to compute relative error
        th = 0.01
        
        for E_out, dot_W_M, col in zip(E_out_arr, dot_W_M_arr, colors):
                            
            err = np.abs( (E_out + dot_W_M) / E_out)                                 
            idx_arr = np.abs(E_out) >= th
                           
            ax1i.semilogy(t[idx_arr], err[idx_arr], '-', c = col)
 
         
        fz = 20
        ax1i.set_xlabel(r'$t$', fontsize = fz)

        cbar = plt.colorbar(sm, ax = ax1i)
        cbar.set_label(label, fontsize = 16)

    ax10.set_ylabel(
        (r'$|(\dot{E}_\mathrm{out} - \dot{W}_\mathrm{M}) / '  
        r'\dot{E}_\mathrm{out}|$'),
        fontsize = fz)

    if show: plt.show()

    fig_path = fig_dir / 'power_rate_of_work_balance_dt_N.png'

    plt.savefig(fig_path)
    
    return
    
    
def plot_energy_rate_for_different_muscle_time_scales(
        h5_filepath,
        show = False):

    '''
    Plot all powers during contraction relaxation experiments 
    '''
    h5, PG = load_h5_file(h5_filepath)

    tau_arr = PG.v_from_key('tau_on') 
    t = h5['t'][:]

    cm = plt.cm.plasma
    norm = mpl.colors.LogNorm(vmin=np.min(tau_arr), vmax=np.max(tau_arr))    
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)    
    colors = sm.to_rgba(tau_arr)
                                        
    # body midpoint index                                        
    mid_idx = int(PG.base_parameter['N_report']/2)
                                           
    k0_arr = h5['CS']['Omega'][:, :, 0, mid_idx]    
    k_arr = h5['FS']['Omega'][:, :, 0, mid_idx]

    powers = EPP.powers_from_h5(h5)
    dot_E_out_arr = h5['powers']['dot_E_out'][:]
    dot_W_M_arr = h5['powers']['dot_W_M'][:]
          
    plt.figure(figsize = (14, 12))     
    gs = plt.GridSpec(3,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
                        
    for k0, k, col in zip(k0_arr, k_arr, colors):
        
        ax0.plot(t, k0, c = col)
        ax1.plot(t, k, c = col)

    fz = 20
    ax0.set_ylabel(r'$k_0$', fontsize = fz)
    ax1.set_ylabel(r'$k$', fontsize = fz)

    ax0.set_xlim([0, 6])
    ax1.set_xlim([0, 6])
               
    th = 0.01
                   
    for dot_E_out, dot_W_M, col in zip(dot_E_out_arr, dot_W_M_arr, colors):
               
        idx_arr = np.abs(dot_E_out) >= th                  
        err = np.abs(dot_E_out + dot_W_M) / np.abs(dot_E_out) 
                        
        ax2.semilogy(t[idx_arr], err[idx_arr], c = col)

    ax2.set_xlim([0, 6])
    
    ax2.set_ylabel(
        (r'$|(\dot{E}_\mathrm{out} - \dot{W}_\mathrm{M}) / '  
        r'\dot{E}_\mathrm{out}|$'),
        fontsize = fz)
    
    ax2.set_xlabel(r'$t$', fontsize = fz)
            
    cbar = plt.colorbar(sm, ax = [ax0, ax1, ax2]) 
    cbar.set_label(r'$\tau_\mathrm{on}$', fontsize = 16)
            
    if show: plt.show() 
    
    fig_path = fig_dir / 'power_rate_of_work_balance_tau.png'
    plt.savefig(fig_path)
    
    return    
    
    
def plot_energy_rate(h5_filepath):
    '''
    Plot all powers during contraction relaxation experiments 
    '''

    # Get simulation results and parameter
    h5_filepath = join(data_path, 'parameter_scans', f'contraction_relaxation_N_dt.h5')
    h5 = h5py.File(h5_filepath, 'r')                           
    PG = ParameterGrid.init_pg_from_filepath(join(log_dir, h5.attrs['grid_filename']))
    
    t = h5['t'][:]
    T = PG.base_parameter['T']
                                           
    k0_1 = h5['CS']['Omega'][-1, :, 0, int(PG.base_parameter['N_report']/2)]    
    k0_1_max = PG.base_parameter['k0'][0]
    
    k1 = h5['FS']['Omega'][-1, :, 0, int(PG.base_parameter['N_report']/2)]
    
    # T0 = Time at which we say that the worm's shape has 
    # converged into the prescribed semicircle configuration
    tol = 0.02
    idx = np.argwhere(np.abs(k1 - k0_1_max) < tol)[0][0]
    T0 = t[idx] 
    
    # T1 = Time at which muscles are switched off and the worm
    # starts to relax back into its straight natural configuration
    T1 = PG.base_parameter['T0'] 

    # T2 = Time at which we say that the worm' shape has converged into 
    # a straight configuration 
    tol = 1e-2
    
    tmp = k0_1.copy()    
    tmp[t <= T1] = k0_1_max        
    idx = np.argwhere(np.abs(tmp) < tol)[0][0]        
    T2 = t[idx] 
         
    FS = h5['FS']        
    dot_V_k = FS['V_dot_k'][-1, :]
    dot_V_sig = FS['V_dot_sig'][-1, :]
    dot_D_k = FS['D_k'][-1, :]
    dot_D_sig = FS['D_sig'][-1, :]
    dot_W_F_lin = FS['dot_W_F_lin'][-1, :]
    dot_W_F_rot = FS['dot_W_F_rot'][-1, :]
    dot_W_M_lin = FS['dot_W_M_lin'][-1, :]
    dot_W_M_rot = FS['dot_W_M_rot'][-1, :]
    
    dot_D = dot_D_k + dot_D_sig + dot_W_F_lin + dot_W_F_rot 
    dot_V = dot_V_k + dot_V_sig    
    dot_W = dot_W_M_lin + dot_W_M_rot 
    
    gs = plt.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
        
    # Plot muscle torque
    ax0.plot(t, k0_1, c = 'k')    
    ax0.plot(t, k1, ls = '--', c = 'k')    
        
    ax0.fill_between([0, T0], 1.1*k0_1_max, color = 'b', alpha = 0.2)
    ax0.fill_between([T0,T1], 1.1*k0_1_max , color = 'r', alpha = 0.2)
    ax0.fill_between([T1,T2], 1.1*k0_1_max, color = 'b', alpha = 0.2)
    ax0.fill_between([T2, T], 1.1*k0_1_max , color = 'r', alpha = 0.2)
    
    ax0.set_xlim([0, T])
    ax0.set_ylim([0, 1.1*k0_1_max])
    
    fz = 20
    
    ax0.set_ylabel(r'$\frac{l_{\mathrm{M}}}{E}$', 
                   rotation = 'horizontal', 
                   fontsize = fz,
                   labelpad = 15.0)
    
    ax1.plot(t, dot_D-dot_V, c = 'k')
    ax1.plot(t, dot_W, ls = '--', c='r')

    ax1.plot([t[0], t[-1]], [0, 0], ls = ':', c = 'k')

    dot_W_max = np.max(dot_W)
    dot_W_min = np.min(dot_W)

    ax1.fill_between([0, T0], 1.1*dot_W_min, 1.2*dot_W_max, color = 'b', alpha = 0.2)
    ax1.fill_between([T0,T1], 1.1*dot_W_min, 1.2*dot_W_max, color = 'r', alpha = 0.2)
    ax1.fill_between([T1,T2], 1.1*dot_W_min, 1.2*dot_W_max, color = 'b', alpha = 0.2)
    ax1.fill_between([T2, T], 1.1*dot_W_min, 1.2*dot_W_max, color = 'r', alpha = 0.2)

    ax1.set_xlabel(r'$t$', fontsize = fz) 
    ax1.set_ylabel(r'$\dot{W}_{\mathrm{M}}$', 
                   fontsize = 18, 
                   rotation = 'horizontal',
                   labelpad = 10.0) 
    
    ax1.set_xlim([0, T])
    ax1.set_ylim([1.1*dot_W_min, 1.2*dot_W_max])
    
    plt.show()

    return
            
def plot_energy_rate_convergence_dt():

    h5_filepath = join(data_path, 'parameter_scans', f'contraction_relaxation_dt.h5')

    h5 = h5py.File(h5_filepath, 'r')                           
    PG = ParameterGrid.init_pg_from_filepath(join(log_dir, h5.attrs['grid_filename']))
                
    v_mat = PG.v_mat
                    
    t  = h5['t'][:]
    T0 = PG.base_parameter['T0']
    T = PG.base_parameter['T']
    
    FS = h5['FS']
    
    k = h5['CS']['Omega']    
    k1 = k[:, 0, 0]
                                
    dot_D_arr, dot_V_arr, dot_W_arr = get_powers_from_h5(h5)
        
    gs = plt.GridSpec(1,1)
    ax00 = plt.subplot(gs[0])
    err_max = 0

    for v, dot_D, dot_V, dot_W_M in zip(v_mat.flatten(), dot_V_arr, dot_D_arr, dot_W_M_arr): 
    
        err = np.abs(dot_D - dot_V + dot_W_M)
    
        if np.max(err) > err_max:
            err_max = np.max(err)
        
        ax00.semilogy(t, err, label = f'dt={v[0]}, N={v[1]}')    
                                                                      
    ax00.fill_between([0, T0], err_max, color = 'b', alpha = 0.2)
    ax00.fill_between([T0, T], err_max, color = 'r', alpha = 0.2)
    ax00.legend()            
                    
    plt.show()
        
    return
            
def plot_energy_rate_for_different_viscosities():
    '''
    Plot all powers during contraction relaxation experiments 
    '''
    
    # Get simulation results and parameter
    h5_filepath = join(data_path, 'parameter_scans', f'contraction_relaxation_eta_nu.h5')
    h5 = h5py.File(h5_filepath, 'r')                           
    PG = ParameterGrid.init_pg_from_filepath(join(log_dir, h5.attrs['grid_filename']))
    
    E = PG.base_parameter['E']    
    nu_arr = np.array([v[0] for v in PG.v_arr])
        
    nu_over_E_arr = nu_arr / E
        
    t = h5['t'][:]
                                           
    k0 = h5['CS']['Omega'][0, :, 0, int(PG.base_parameter['N_report']/2)]    

    k_arr = h5['FS']['Omega'][:, :, 0, int(PG.base_parameter['N_report']/2)]    

    dot_D_arr, dot_V_arr, dot_W_arr = get_powers_from_h5(h5)
     
    gs = plt.GridSpec(3,1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
        
    ax0.plot(t, k0, c = 'k')
                
    for nu_over_E, k in zip(nu_over_E_arr, k_arr):
        
        ax0.plot(t, k, label = nu_over_E)
                
    ax0.legend()

    for dot_D, dot_V, dot_W in zip(dot_D_arr, dot_V_arr, dot_W_arr):
                
        l = ax1.plot(t, dot_D - dot_V, ls = '-')
        ax1.plot(t, dot_W, ls = '--', c = l[0].get_color()) 
                
    for dot_D, dot_V, dot_W in zip(dot_D_arr, dot_V_arr, dot_W_arr):
                
        err = np.abs(dot_D - dot_V - dot_W) 
        
        ax2.plot(t, err)
    
    plt.show()    

if __name__ == '__main__':
                
#     plot_energy_conservation_for_different_dt_and_N(
# 'contraction_relaxation_dt=0.0001_dt_max=0.01_N=125_tau_on=0.10_k0=3.14_pi=False.h5',
# 'contraction_relaxation_N_min=125_N_max=2000_dt=0.0001_tau_on=0.10_k0=3.14_pi=False.h5',        
#         show = False)

    plot_energy_rate_for_different_muscle_time_scales(
'contraction_relaxation_tau_min=0.01_tau_max=0.2_tau_step=0.01_k0=3.14_mu=1.0_N=100_dt=0.001.h5',
        show = False)        
                
    #plot_energy_rate()    
    #plot_energy_rate_convergence()
    #plot_energy_rate_for_different_viscosities()
    #plot_energy_rate_for_different_muscle_time_scales()
    #plot_energy_rate_convergence_dt()
    
    print('Finished')
    
    
    

    


