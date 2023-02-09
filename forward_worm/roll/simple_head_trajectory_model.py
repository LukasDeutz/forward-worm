'''
Created on 5 Feb 2023

@author: lukas

This is a simple model to understand the shape of projected head point 
trajectories during roll maneuvers.

The head point is assumed to move in a circle in a rotating 
body frame with a fixed origin. 

The rotation of the frame mimics the worm's rolling whereas 
the circular motion mimics the change in head position due to
circular muscle activations in the worm's neck. 
'''
import numpy as np
import matplotlib.pyplot as plt

def comput_head_trajectory(f_M, f_R, T=None):
        
    w_M = 2*np.pi*f_M
    w_R = 2*np.pi*f_R
    
    if T is None: T=5/f_M
            
    t = np.linspace(0, T, int(1e3))
    
    x = np.cos(w_M*t)*np.cos(w_R*t) + np.sin(w_M*t)*np.sin(w_R*t)
    y = -np.cos(w_M*t)*np.sin(w_R*t) + np.sin(w_M*t)*np.cos(w_R*t)
    
    return x, y

def test_plot_head_trajectory():
    
    f_M = 1.0 # Muscle frequency
    f_R = -0.5 # Roll frequency
        
    x, y = comput_head_trajectory(f_M, f_R)

    plt.plot(x, y)
    plt.show()

    return

def plot_special_cases():
    
    f_M = 1.0 # Muscle frequency
    f_R = 1.0 # Roll frequency
    
    x, y = comput_head_trajectory(f_M, f_R, T=None)

    N = 3 # Number cases
    plt.figure(figsize = (6*N, 6))            
    gs = plt.GridSpec(1, N)    
    fz = 20
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(x, y, 'o-')
    ax0.set_title('$f_\mathrm{M}=f_\mathrm{R}$', fontsize = fz)
    ax0.set_ylabel('$y$', fontsize = fz)    
    ax0.set_xlabel('$x$', fontsize = fz)    
    ax0.set_xlim((-1.1, 1.1))
    ax0.set_ylim((-1.1, 1.1))
    ax0.set_aspect('equal')
    
    f_M = 1.0 # Muscle frequency
    f_R = 0.0 # Roll frequency
    x, y = comput_head_trajectory(f_M, f_R, T=None)

    ax1 = plt.subplot(gs[1])
    ax1.plot(x, y, '-')
    ax1.set_title('$f_\mathrm{R}=0$', fontsize = fz)
    ax1.set_xlabel('$x$', fontsize = fz)    
    ax1.set_xlim((-1.1, 1.1))
    ax1.set_ylim((-1.1, 1.1))
    ax1.set_aspect('equal')

    f_M = 1.0 # Muscle frequency
    f_R = 0.9 # Roll frequency
    
    x, y = comput_head_trajectory(f_M, f_R, T=None)

    ax2 = plt.subplot(gs[2])
    ax2.plot(x, y, '-')
    ax2.set_title('$f_\mathrm{R} / f_\mathrm{M} = $' + f'{f_R/f_M:.2f}', fontsize = fz)
    ax2.set_xlabel('$x$', fontsize = fz)    
    ax2.set_xlim((-1.1, 1.1))
    ax2.set_ylim((-1.1, 1.1))
    ax2.set_aspect('equal')
    
    plt.show()
    
    return
    
def plot_trajectory_shapes():
    
    N = 3 # number of plots
    f_M = 5.0 # Muscle frequency
    c = np.linspace(0, 0.1, N)
    f_R_arr = c * f_M # Roll frequency

    T = 5/f_M 

    for f_R in f_R_arr:
        x, y = comput_head_trajectory(f_M, f_R, T)
    
        plt.plot(x, y)
    
    plt.show()
    
    return


if __name__ == '__main__':
    
    #plot_trajectory_shapes()    
    #plot_special_cases()
    
    test_plot_head_trajectory()
    








    
    
    
    
    
    






