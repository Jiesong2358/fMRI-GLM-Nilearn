import os
import pandas as pd
import nibabel as nib
from nilearn.image import load_img, mean_img, threshold_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from nilearn.masking import compute_epi_mask
from nilearn import image

# Set subject range
subject_range = range(1, 50)

for sub in subject_range:
    output_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/Imag_RSA_6cons/sub-{:0>2d}'.format(sub)
    if sub in [1,2,3,4,6,11,13,14,23,36,48]:
        continue
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Set root directories
    root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new/derivatives'
    events_root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new'
    # Load template and set TR
    template = 'C:\\Jie_Documents\\Navigation\\Script\\Nilearn_GLM\\Atlas\\tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
    tr = 1.3
    # Accumulate design matrices and fMRI data across all runs
    all_design_matrices = []
    all_fmri_data = []
    
    for run in range(1, 7):
        run_file = os.path.join(root_dir, 'sub-{:0>2d}/func/sub-{:0>2d}_task-int3_run-{:0>2d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, sub, run))
        #if no file named sub-{:0>2d}_run-{:0>2d}_smoothed.nii.gz found, then smooth the run_file, otherwise continue
        if not os.path.exists(os.path.join(output_dir, 'sub-{:0>2d}_run-{:0>2d}_smoothed.nii.gz'.format(sub, run))):
            run_smoothed = image.smooth_img(run_file, fwhm=6)
            #save the smoothed image
            smoothed_file = os.path.join(output_dir, 'sub-{:0>2d}_run-{:0>2d}_smoothed.nii.gz'.format(sub, run))
            run_smoothed.to_filename(smoothed_file)
        run_smoothed = image.smooth_img(run_file, fwhm=6)
        img = nib.load(run_file)
        num_scans = img.shape[3]
        frame_times = np.arange(num_scans) * tr
        ## load smoothed image
        fmri_img = run_smoothed
        ## Load no smoothing image:
        # fmri_img = load_img(run_file)
        mean_img_ = mean_img(fmri_img)
        mask = compute_epi_mask(mean_img_)
        # If file named full_brain_mask_run{:0>2d}.nii.gz not found, then threshold the mask, otherwise continue
        if not os.path.exists(os.path.join(output_dir, 'full_brain_mask_run{:0>2d}.nii.gz'.format(run))):
            binary_mask = threshold_img(mask, threshold=0.5)
            mask_file = os.path.join(output_dir, 'full_brain_mask_run{:0>2d}.nii.gz'.format(run))
            binary_mask.to_filename(mask_file)
        mask_file = os.path.join(output_dir, 'full_brain_mask_run{:0>2d}.nii.gz'.format(run))
        full_mask = load_img(mask_file)
        events_file = os.path.join(events_root_dir,'sub-{:0>2d}/func/sub-{:0>2d}_task-rep_ret_run-{:0>2d}_imagination.tsv'.format(sub, sub, run))
        events_data = pd.read_csv(events_file, sep="\t")
        confound_file = os.path.join(root_dir,'sub-{:0>2d}/func/sub-{:0>2d}_task-int3_run-{:0>2d}_desc-confounds_timeseries.tsv'.format(sub, sub, run))
        selected_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                        'global_signal', 
                        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'a_comp_cor_06', 'a_comp_cor_07', 'a_comp_cor_08', 'a_comp_cor_09',]
        motion = pd.read_csv(confound_file, sep="\t", usecols=selected_columns)
        hrf_model = "spm"
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events_data,
            drift_model="polynomial",
            drift_order=3,
            add_regs=motion,
            hrf_model=hrf_model,
        )
        # Plot the design matrix and save it to file
        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_design_matrix(design_matrix, ax=ax)
        ax.set_title("sub-{:0>2d}_run{:0>2d}".format(sub, run), fontsize=12)
        output_file = os.path.join(output_dir, 'Imagination_design_matrix_run{:0>2d}.png'.format(run))
        # Save the design matrix plot to file
        plt.savefig(output_file)
        # Append the design matrix and fMRI data  
        all_design_matrices.append(design_matrix)
        all_fmri_data.append(fmri_img)
    
    # Concatenate design matrices and fMRI data
    design_matrices_concatenated = pd.concat(all_design_matrices, axis=0)
    fmri_data_concatenated = concat_imgs(all_fmri_data)
    
    # Fit the GLM model with the accumulated data
    fmri_glm = FirstLevelModel(mask_img=full_mask, minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_data_concatenated, design_matrices=design_matrices_concatenated)

    # Define the contrasts
    n_columns = len(design_matrices_concatenated.columns)
    def pad_vector(contrast_, n_columns):
        """Append zeros in contrast vectors."""
        return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))
    
    # Define the contrasts
    # contrasts = {
    #     'sub-{:0>2d}_Imag_allo_repe-ego_repe'.format(sub): pad_vector([1, 0, -1, 0],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-ego_retra'.format(sub): pad_vector([0, 1, 0, -1],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-allo_repe'.format(sub): pad_vector([-1, 1, 0, 0],n_columns),
    #     'sub-{:0>2d}_Imag_ego_retra-ego_repe'.format(sub): pad_vector([0, 0, -1, 1],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-ego_repe'.format(sub): pad_vector([0, 1, -1, 0],n_columns),
    #     'sub-{:0>2d}_Imag_allo-ego'.format(sub): pad_vector([1, 1, -1, -1],n_columns),
    #     'sub-{:0>2d}_Imag_retra-repe'.format(sub): pad_vector([-1, 1, -1, 1],n_columns),
    # }
    
    contrasts = {
        'sub-{:0>2d}_Imag_repe_car'.format(sub): pad_vector([1, 0, 0, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_repe_int1'.format(sub): pad_vector([0, 1, 0, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_int1'.format(sub): pad_vector([0, 0, 1, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_repe_int2'.format(sub): pad_vector([0, 0, 0, 1, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_int2'.format(sub): pad_vector([0, 0, 0, 0, 1, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_phone'.format(sub): pad_vector([0, 0, 0, 0, 0, 1],n_columns),
        }
    print('Computing contrasts...')
    # Estimate the contrasts. Note that the model implicitly computes a fixed
    # effect across the six runs

    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id))
        # Estimate the contrasts. Note that the model implicitly computes a fixed
        # effect across the six runs
        t_map = fmri_glm.compute_contrast(
            contrast_val, 
            stat_type ='t',                      
            output_type='stat')
        # write the resulting stat images to file
        t_image_path = os.path.join(output_dir, f'{contrast_id}_t_map.nii.gz')
        t_map.to_filename(t_image_path)
        # plot the t_map of all the contrasts
        plotting.plot_stat_map(
            t_map, threshold=3.0, 
            display_mode='ortho', colorbar=True,
            draw_cross=False, 
            title=f'{contrast_id}, fixed effects')
        output_file = os.path.join(output_dir, f't_map_{contrast_id}.png')
        plt.savefig(output_file)
        # Obtain z map
    # contrasts1 = {
    #     'sub-{:0>2d}_Imag_allo_repe-ego_repe'.format(sub): pad_vector([1, 0, -1, 0],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-ego_retra'.format(sub): pad_vector([0, 1, 0, -1],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-allo_repe'.format(sub): pad_vector([-1, 1, 0, 0],n_columns),
    #     'sub-{:0>2d}_Imag_ego_retra-ego_repe'.format(sub): pad_vector([0, 0, -1, 1],n_columns),
    #     'sub-{:0>2d}_Imag_allo_retra-ego_repe'.format(sub): pad_vector([0, 1, -1, 0],n_columns),
    #     #fixed effects for 6 runs-effect of interest
    #     'Effects_of_interest': np.eye(n_columns)[:4]
    # }
    contrasts1 = {
        'sub-{:0>2d}_Imag_repe_car'.format(sub): pad_vector([1, 0, 0, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_repe_int1'.format(sub): pad_vector([0, 1, 0, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_int1'.format(sub): pad_vector([0, 0, 1, 0, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_repe_int2'.format(sub): pad_vector([0, 0, 0, 1, 0, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_int2'.format(sub): pad_vector([0, 0, 0, 0, 1, 0],n_columns),
        'sub-{:0>2d}_Imag_retra_phone'.format(sub): pad_vector([0, 0, 0, 0, 0, 1],n_columns),
        #fixed effects for 6 runs-effect of interest
        'Effects_of_interest': np.eye(n_columns)[:4]
    }
    print('Computing contrasts...')
    # Obtain z map
    for index, (contrast_id, contrast_val) in enumerate(contrasts1.items()):
        print('  Contrast % 2i out of %i: %s' % (
            index + 1, 
            len(contrasts1), 
            contrast_id))
        # Estimate the contasts. Note that the model implicitly computes a fixed
        # effect across the six runs
        z_map = fmri_glm.compute_contrast(
            contrast_val, output_type='z_score')
        # write the resulting stat images to file
        z_image_path = os.path.join(output_dir, f'{contrast_id}_z_map.nii.gz')
        z_map.to_filename(z_image_path)
        # plot the z_map of all the contrasts
        plotting.plot_stat_map(
            z_map, threshold=3.0, 
            display_mode='ortho', colorbar=True,
            draw_cross=False, 
            title=f'{contrast_id}, fixed effects')
        output_file = os.path.join(output_dir, f'z_map_{contrast_id}.png')
        plt.savefig(output_file)