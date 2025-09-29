import os
import pandas as pd
import nibabel as nib
from nilearn.image import load_img, mean_img, threshold_img, concat_imgs, smooth_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from nilearn.masking import compute_epi_mask

# Set subject range (excluding certain subjects)
subject_range = [sub for sub in range(1, 50) if sub not in [2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 17, 18, 22, 28, 34, 36]]

# Set root directories
root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new/derivatives'
events_root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new'
template = 'C:\\Jie_Documents\\Navigation\\Script\\Nilearn_GLM\\Atlas\\tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
tr = 1.3  # Repetition time

def process_subject(sub):
    output_dir = f'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/1Inter_retrieval_allo-ego/sub-{sub:02d}'
    os.makedirs(output_dir, exist_ok=True)

    all_design_matrices = []
    all_fmri_data = []

    for run in range(1, 5):
        run_file = os.path.join(root_dir, f'sub-{sub:02d}/func/sub-{sub:02d}_task-int1_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        smoothed_file = os.path.join(root_dir, f'sub-{sub:02d}/func/sub-{sub:02d}_task-int1_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold_fwhm6.nii.gz')
        # Apply smoothing if the file doesn't exist
        if not os.path.exists(smoothed_file):
            smoothed_img = smooth_img(run_file, fwhm=6)
            smoothed_img.to_filename(smoothed_file)
        else:
            smoothed_img = load_img(smoothed_file)

        # Compute mask
        mean_img_ = mean_img(smoothed_img)
        mask = compute_epi_mask(mean_img_)
        mask_file = os.path.join(root_dir, f'sub-{sub:02d}/func/full_brain_mask_task-int1_run{run:02d}.nii.gz')

        if not os.path.exists(mask_file):
            binary_mask = threshold_img(mask, threshold=0.5)
            binary_mask.to_filename(mask_file)
        full_mask = load_img(mask_file)

        # Load events and confounds
        events_file = os.path.join(events_root_dir, 
                                   f'sub-{sub:02d}/func/sub-{sub:02d}_task-dir_rep1_run-{run:02d}_ego_allo.tsv')
        confound_file = os.path.join(root_dir, 
                                     f'sub-{sub:02d}/func/sub-{sub:02d}_task-int1_run-{run:02d}_desc-confounds_timeseries.tsv')

        motion_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal'] + [f'a_comp_cor_0{i}' for i in range(10)]
        motion = pd.read_csv(confound_file, sep="\t", usecols=motion_columns)

        events_data = pd.read_csv(events_file, sep="\t")
        # rename the events file
        num_scans = nib.load(run_file).shape[3]
        frame_times = np.arange(num_scans) * tr
        
        # Create design matrix
        design_matrix = make_first_level_design_matrix(frame_times, events_data, drift_model="polynomial", drift_order=3, 
                                                       add_regs=motion, hrf_model="spm")
        all_design_matrices.append(design_matrix)
        all_fmri_data.append(smoothed_img)

        # Save design matrix plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_design_matrix(design_matrix, ax=ax)
        ax.set_title(f"sub-{sub:02d}_run{run:02d}", fontsize=12)
        plt.savefig(os.path.join(output_dir, f'Imagination_design_matrix_run{run:02d}.png'))
        plt.close()

    # Fit GLM model
    fmri_data_concatenated = concat_imgs(all_fmri_data)
    design_matrices_concatenated = pd.concat(all_design_matrices, axis=0)

    fmri_glm = FirstLevelModel(mask_img=full_mask, minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_data_concatenated, design_matrices=design_matrices_concatenated)

    # Define contrasts
    n_columns = len(design_matrices_concatenated.columns)

    def pad_vector(contrast_, n_columns):
        """Append zeros in contrast vectors."""
        return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


    contrasts = {
        'Retrieval_allo-ego': pad_vector([1, -1], n_columns),
        # 'Encoding_allo_NE_NW': pad_vector([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_allo_NE_SE': pad_vector([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_allo_NE_SW': pad_vector([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_allo_SE_NW': pad_vector([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_allo_SE_SW': pad_vector([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_allo_SW_NW': pad_vector([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_NE_NW': pad_vector([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_NE_SE': pad_vector([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_NE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_SE_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_SE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Encoding_ego_SW_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),

        # 'Retrieval_allo_NE_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_allo_NE_SE': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_allo_NE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_allo_SE_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_allo_SE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_allo_SW_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_ego_NE_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], n_columns),
        # 'Retrieval_ego_NE_SE': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], n_columns),
        # 'Retrieval_ego_NE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], n_columns),
        # 'Retrieval_ego_SE_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], n_columns),
        # 'Retrieval_ego_SE_SW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], n_columns),
        # 'Retrieval_ego_SW_NW': pad_vector([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], n_columns),
    }

    # Compute & save t-maps and z-maps
    for contrast_id, contrast_val in contrasts.items():
        print(f"Computing {contrast_id} for sub-{sub:02d}...")

        # Compute t-map
        t_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='stat')
        t_map.to_filename(os.path.join(output_dir, f'sub-{sub:02d}_{contrast_id}_t_map.nii.gz'))

        # Compute z-map
        z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
        z_map.to_filename(os.path.join(output_dir, f'sub-{sub:02d}_{contrast_id}_z_map.nii.gz'))

        # Plot t-map
        plotting.plot_stat_map(t_map, threshold=3.0, display_mode='ortho', colorbar=True, 
                               draw_cross=False, title=f'{contrast_id} - T map')
        plt.savefig(os.path.join(output_dir, f't_map_{contrast_id}.png'))
        plt.close()

        # Plot z-map
        plotting.plot_stat_map(z_map, threshold=3.0, display_mode='ortho', colorbar=True, 
                               draw_cross=False, title=f'{contrast_id} - Z map')
        plt.savefig(os.path.join(output_dir, f'z_map_{contrast_id}.png'))
        plt.close()

# Run for all subjects
for sub in subject_range:
    process_subject(sub)

