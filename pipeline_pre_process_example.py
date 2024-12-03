import pre_process_mm


if __name__ == '__main__':


	root_dir = '/Users/noravivancogonzalez/Desktop/20241114'

	drift_corrected_path = pre_process_mm.drift_correct(root_dir, 'DuMM', c=0)

	path_to_rotated_images = pre_process_mm.rotate_stack(drift_corrected_path, c=0, growth_channel_length=400)
	# growth channel length depends on # of pixels, which is affected by binning
	# had to change rotation angle to not be negative since growth channels were facing downward

	#path_to_hyperstacked = '/Users/noravivancogonzalez/Desktop/20241001/dimm_CL000_1/only_analyze_these/hyperstacked/drift_corrected/rotated'

	#pre_process_mm.unstack_tcyx_to_cyx(path_to_hyperstacked)

	#path_to_rotated_images = '/Users/noravivancogonzalez/Desktop/20241114/DuMM_CL001_CL006_glucose/hyperstacked/drift_corrected/rotated'

	path_to_mm_channels = pre_process_mm.extract_mm_channels(path_to_rotated_images)


	FOV = '003'
	empty_stack_id = '692'
	ana_peak_ids = ['41', '68', '95']
	phase_channel = 0

	#pre_process_mm.subtract_fov_stack(path_to_mm_channels, FOV, empty_stack_id, ana_peak_ids, phase_channel)