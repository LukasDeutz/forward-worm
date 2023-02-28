'''
Created on 10 Feb 2023

@author: lukas
'''
from pathlib import Path

data_dir = Path('../../data/contraction_relaxation')
log_dir = data_dir / 'logs'
sim_dir = data_dir / 'simulations'
sweep_dir = data_dir / 'parameter_sweeps'
fig_dir = Path('../../figures/contraction_relaxation')
video_dir = Path('../../videos/contraction_relaxation')

if not data_dir.is_dir(): data_dir.mkdir(parents = True, exist_ok = True)
if not sim_dir.is_dir(): sim_dir.mkdir(exist_ok = True)
if not log_dir.is_dir(): log_dir.mkdir(exist_ok = True)
if not sweep_dir.is_dir(): sweep_dir.mkdir(exist_ok = True)
if not fig_dir.is_dir(): fig_dir.mkdir(parents = True, exist_ok = True)
if not video_dir.is_dir(): video_dir.mkdir(parents = True, exist_ok=True)

