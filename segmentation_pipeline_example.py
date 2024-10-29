import mm3_adapted_cell_segmentation as mm3_seg
path_to_stack = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1/hyperstacked/drift_corrected/rotated/mm_channels/subtracted_phase/phase_subtracted_FOV003_region_95.tif'
celltk_path_out = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1/hyperstacked/drift_corrected/rotated/mm_channels/subtracted_phase/FOV_003/channel_95/outputs'
mm3_seg.segment_chnl_stack(path_to_stack, celltk_path_out)

labeled_stack = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1/hyperstacked/drift_corrected/rotated/mm_channels/subtracted_phase/mm3_segmented_phase_subtracted_FOV003_region_95.tif'
mm3_seg.display_segmentation(path_to_stack, labeled_stack, start = 0, end = 20)