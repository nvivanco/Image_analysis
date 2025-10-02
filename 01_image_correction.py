import argparse
import os
from mmtrack import pre_process_mm


def run_image_correction(input_dir, output_dir, phase_channel_idx,
						 fast_drift_correction, growth_len, trench_ends_orientation):
	"""
	Performs drift correction, image rotation, and channel extraction on raw DuMM TIFF stacks.
	"""
	print(f"Starting image pre-processing for raw data at: {input_dir}")
	os.makedirs(output_dir, exist_ok=True)

	# 1. DRIFT CORRECTION
	print("\n--- 1. Performing Drift Correction ---")
	drift_corrected_path = pre_process_mm.drift_correct(
		dir_files_to_correct=input_dir,
		experiment_type='DuMM',
		fast4=fast_drift_correction,
		c=phase_channel_idx  # Phase channel index
	)
	print(f"Drift correction complete. Files path: {drift_corrected_path}")

	# 2. IMAGE ROTATION
	print("\n--- 2. Performing Image Rotation ---")
	path_to_rotated_images = pre_process_mm.rotate_stack(
		drift_corrected_path,
		c=phase_channel_idx,  # Phase channel index
		growth_channel_length=growth_len,
		closed_ends=trench_ends_orientation
	)
	print(f"Rotation complete. Files path: {path_to_rotated_images}")

	# 3. CHANNEL EXTRACTION
	print("\n--- 3. Extracting Mother Machine Channels ---")
	path_to_mm_channels = pre_process_mm.extract_mm_channels(path_to_rotated_images)
	print(f"Channel extraction complete. Files path: {path_to_mm_channels}")

	print("\n Pipeline Stage 1 (Image Correction) Complete.")
	print("\n\n#################################################################")
	print("## USER ACTION REQUIRED: Inspect TRENCH MASKS and TIME-LAPSE   ##")
	print(f"## Next Step: Determine 'empty_stack_id' and 'ana_peak_ids'. ##")
	print(f"## Use {path_to_mm_channels} as the input for Stage 2.         ##")
	print("#################################################################")

	return path_to_mm_channels


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Stage 1: Drift correction, rotation, and channel extraction for DuMM time-lapse images."
	)

	# --- Mandatory I/O Arguments ---
	parser.add_argument(
		'--input-dir',
		required=True,
		type=str,
		help="Path to the directory containing the raw TIFF stacks."
	)
	parser.add_argument(
		'--output-dir',
		required=True,
		type=str,
		help="Base path for saving corrected and processed images."
	)

	# --- Optional Function Parameters (Defaults set here) ---
	parser.add_argument(
		'--phase-channel-idx',
		type=int,
		default=0,
		help="Index of the phase contrast channel (c parameter in functions). Default: 0."
	)
	parser.add_argument(
		'--fast-drift-correction',
		type=bool,
		default=True,
		help="Use the fast4 drift correction method (fast4 parameter). Default: True."
	)
	parser.add_argument(
		'--growth-channel-length',
		type=int,
		default=400,
		help="Approx. pixel length for 1x1 binning in DuMM (growth_channel_length). Default: 400."
	)
	parser.add_argument(
		'--trench-ends-orientation',
		type=str,
		default='down',
		choices=['up', 'down'],
		help="Orientation of the closed trench ends ('closed_ends' parameter). Default: 'down'."
	)

	args = parser.parse_args()
	run_image_correction(
		input_dir=args.input_dir,
		output_dir=args.output_dir,
		phase_channel_idx=args.phase_channel_idx,
		fast_drift_correction=args.fast_drift_correction,
		growth_len=args.growth_channel_length,
		trench_ends_orientation=args.trench_ends_orientation
	)