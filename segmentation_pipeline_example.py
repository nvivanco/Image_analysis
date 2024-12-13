import mm3_adapted_cell_segmentation as mm3_seg

path_to_stack = '/Users/noravivancogonzalez/Desktop/20241114/hyperstacked/drift_corrected/rotated/mm_channels/subtracted//subtracted_FOV_001_region_1164_c_0.tif'
output_path = '/Users/noravivancogonzalez/Desktop/20241114/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/outputs'
mm3_seg.segment_chnl_stack(path_to_stack,
						   output_path,
						   OTSU_threshold=1.5,
						   first_opening=5,
						   distance_threshold=3,
						   second_opening_size=1,
						   min_object_size=5) # depends on pixels and whether there is binning, for 3x3 use 25




#labeled_stack = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1/hyperstacked/drift_corrected/rotated/mm_channels/subtracted_phase/mm3_segmented_phase_subtracted_FOV003_region_95.tif'
#mm3_seg.display_segmentation(path_to_stack, labeled_stack, start = 0, end = 20)