import pre_process_mm


if __name__ == '__main__':


	#root_dir = '/Users/noravivancogonzalez/Desktop/20241114'

	#drift_corrected_path = pre_process_mm.drift_correct(root_dir, 'DuMM', c=0)

	#path_to_rotated_images = pre_process_mm.rotate_stack(drift_corrected_path, c=0, growth_channel_length=400)
	# growth channel length depends on # of pixels, which is affected by binning
	# had to change rotation angle to not be negative since growth channels were facing downward

	#path_to_mm_channels = pre_process_mm.extract_mm_channels(path_to_rotated_images)

	path_to_mm_channels = '/Users/noravivancogonzalez/Desktop/20241114/hyperstacked/drift_corrected/rotated/mm_channels'


	FOV = '001'
	empty_stack_id = '707'
	ana_peak_ids = ['1164', '1314', '2227']
	channel = 1

	pre_process_mm.subtract_fov_stack(path_to_mm_channels, FOV, empty_stack_id, ana_peak_ids, method = 'fluor', channel_index= channel)

	#I see a lot of drift at t=82

	#path_to_hyperstacked = '/Users/noravivancogonzalez/Desktop/20241114/hyperstacked/drift_corrected/rotated'

	#pre_process_mm.unstack_tcyx_to_cyx(path_to_hyperstacked)