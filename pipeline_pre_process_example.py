import pre_process_mm


if __name__ == '__main__':
	root_dir = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1'

	drift_corrected_path = pre_process_mm.napari_ready_format_drift_correct(root_dir, 'dimm', c = 0)
	drift_corrected_path = '/Users/noravivancogonzalez/Desktop/20240919/dimm_giTG66_glucose_1/hyperstacked/drift_corrected'
	path_to_rotated_images = pre_process_mm.rotate_stack(drift_corrected_path, c=0, growth_channel_length=280)
	path_to_mm_channels = pre_process_mm.extract_mm_channels(path_to_rotated_images)

	FOV = '003'
	empty_stack_id = '692'
	ana_peak_ids = ['41', '68', '95']
	phase_channel = 0

	pre_process_mm.subtract_fov_stack(path_to_mm_channels, FOV, empty_stack_id, ana_peak_ids, phase_channel)