# Third-party imports
import h5py
from pathlib import Path

# Local imports
from forward_worm.forward_undulation.undulations_dirs import data_path, log_dir, sim_dir, sweep_dir, video_path
from parameter_scan import ParameterGrid
from simple_worm_experiments.worm_studio import WormStudio
from simple_worm_experiments.experiment_post_processor import EPP
from simple_worm.plot3d_cosserat import plot_CS_vs_FS, plot_multiple_scalar_fields

def generate_worm_videos(h5_filename):

    h5_filepath = sweep_dir / h5_filename
    h5 = h5py.File(str(h5_filepath), 'r')    
    PG_filepath = log_dir / h5.attrs['grid_filename']
    PG = ParameterGrid.init_pg_from_filepath(str(PG_filepath))

    vid_filenames = [f'A={p["A"]}_lam={p["lam"]}_f_{p["f"]}_'
        f'_mu_{p["mu"]:.2f}_N_{p["N"]}_dt_{p["dt"]}' for p in PG.param_arr]

    output_dir = video_path / Path(h5_filename).stem
    if not output_dir.exists(): output_dir.mkdir()
                                                                                             
    WormStudio.generate_worm_clips_from_PG(
        PG, vid_filenames, output_dir, sim_dir, 
        add_trajectory = False,
        draw_e3 = False,
        n_arrows = 0.2)

    return

if __name__ == '__main__':
        
    generate_worm_videos(
'undulation_rft_A_min=1.0_A_max=8.0_A_step=1.0_f=2.0_mu=0.001_N=100_dt=0.01.h5')
                        
