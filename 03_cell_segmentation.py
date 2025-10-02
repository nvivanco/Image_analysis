import argparse
import os
import json
import tifffile as tf
from mmtrack import cell_segmentation, plot_cells




def get_tiff_frame_count(file_path):
	"""Reads a TIFF file and returns the index of the last time frame (T - 1)."""
	try:
		# 1. Read the image data into a NumPy array
		img_stack = tf.imread(file_path)

		# 2. Get the shape of the array
		shape = img_stack.shape

		# 3. Assume the time axis (T) is the first dimension
		if len(shape) >= 3:
			# The last index is T - 1, which is shape[0] - 1
			last_frame_index = shape[0] - 1
			return last_frame_index
		else:
			# Handle case where the file might be a single 2D image (T=1, last index 0)
			return 0

	except FileNotFoundError:
		print(f"Error: TIFF file not found at {file_path}")
		return 0
	except Exception as e:
		print(f"Error reading TIFF file {file_path}: {e}")
		return 0



def run_cell_segmentation(base_dir, exp_dict_json, start_frame, end_frame,
						  phase_c_str, otsu_thresh,
						  first_open, dist_thresh, second_open_size,
						  min_area, max_area, small_merge_thresh):
	"""
	Performs cell segmentation on multiple subtracted stacks.
	"""

	try:
		seg_FOV_dict = json.loads(exp_dict_json)
	except json.JSONDecodeError:
		print("ERROR: Could not parse the experiment dictionary. Ensure it is valid JSON.")
		return

	print(f"Starting cell segmentation across {len(seg_FOV_dict)} experiments.")

	# Flag to check if we've determined the frame count yet
	default_end_frame_is_set = (end_frame is not None)

	for folder, fov_dict in seg_FOV_dict.items():
		print(f"\nProcessing Experiment: {folder}")
		for fov_id in fov_dict.keys():
			ana_peak_ids = fov_dict[fov_id]
			print(f"  FOV: {fov_id}, Trenches: {ana_peak_ids}")

			for peak_id in ana_peak_ids:
				base_file_path = os.path.join(base_dir, folder, 'hyperstacked', 'drift_corrected', 'rotated',
											  'mm_channels', 'subtracted')

				path_to_phase_stack = os.path.join(base_file_path,
												   f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif')
				labeled_stack = os.path.join(base_file_path,
											 f'mm3_segmented_subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif')
				output_path = os.path.join(base_file_path, 'outputs')

				os.makedirs(output_path, exist_ok=True)

				# Only calculate if the user didn't provide a value and it's the first time
				current_end_frame = end_frame
				if current_end_frame is None:
					current_end_frame = get_tiff_frame_count(path_to_phase_stack)
					if current_end_frame == 0:
						print(f"WARNING: Could not determine frame count for {peak_id}. Skipping.")
						continue
					print(
						f"    -> Dynamically set end_frame to {current_end_frame} (Total frames: {current_end_frame + 1}).")

				print(f"    -> Segmenting Trench {peak_id}...")

				# --- SEGMENTATION ---
				cell_segmentation.segment_chnl_stack(
					path_to_phase_stack,
					output_path,
					OTSU_threshold=otsu_thresh,
					first_opening=first_open,
					distance_threshold=dist_thresh,
					second_opening_size=second_open_size,
					min_cell_area=min_area,
					max_cell_area=max_area,
					small_merge_area_threshold=small_merge_thresh
				)

				# --- VISUAL CONFIRMATION ---
				print(f"    -> Displaying results for visual confirmation...")
				plot_cells.display_segmentation(
					path_to_phase_stack,
					mask_path=labeled_stack,
					start=start_frame,
					end=current_end_frame,  # Use the determined frame index
					alpha=0.5
				)

	print("\nPipeline Stage 3 (Cell Segmentation) Complete.")
	print("\n✅ Pipeline Stage 3 (Cell Segmentation) Complete.")
	print("\n\n#################################################################")
	print("# USER ACTION REQUIRED: Visually inspect segmentation results.")
	print("# Adjust segmentation parameters if cell boundaries are wrong.")
	print("#################################################################")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Stage 3: Cell segmentation using highly tunable parameters for cell_segmentation."
	)

	# --- Mandatory I/O Argument ---
	parser.add_argument(
		'--base-dir',
		required=True,
		type=str,
		help="Base path containing all experiment folders (e.g., /path/to/DuMM_image_analysis)."
	)
	parser.add_argument(
		'--exp-dict',
		required=True,
		type=str,
		help='JSON string defining experiments/FOVs/trenches. Format: {"exp_dir": {"FOV": ["peak_id", ...]}}.'
	)

	# --- Time Frame Arguments ---
	parser.add_argument('--start-frame', type=int, default=0, help="Starting time frame for segmentation. Default: 0.")
	parser.add_argument(
		'--end-frame',
		type=int,
		default=None,  # Crucially, set default to None
		help="Last time frame index to consider. If omitted, it is determined dynamically from the TIFF file size."
	)

	# --- Phase Channel Index Arguments (as strings) ---
	parser.add_argument('--phase-channel', type=str, default='0', help="Phase channel index as string. Default: '0'.")

	# --- Segmentation Hyperparameters (Tunable) ---
	parser.add_argument('--otsu-threshold', type=float, default=0.5,
						help="OTSU_threshold for segmentation (0.5 to 1.5). Default: 0.5.")
	parser.add_argument('--first-opening', type=int, default=4,
						help="Size of the first morphological opening. Default: 4.")
	parser.add_argument('--distance-threshold', type=float, default=3.0,
						help="Distance threshold for watershed seeding. Default: 3.0.")
	parser.add_argument('--second-opening-size', type=int, default=3,
						help="Size of the second morphological opening. Default: 3.")
	parser.add_argument('--min-cell-area', type=int, default=100, help="Minimum cell area in pixels. Default: 100.")
	parser.add_argument('--max-cell-area', type=int, default=1000, help="Maximum cell area in pixels. Default: 1000.")
	parser.add_argument('--small-merge-area-threshold', type=int, default=100,
						help="Threshold for merging small segmented regions. Default: 100.")

	args = parser.parse_args()

	run_cell_segmentation(
		base_dir=args.base_dir,
		exp_dict_json=args.exp_dict,
		start_frame=args.start_frame,
		end_frame=args.end_frame,  # Passes None if not provided
		phase_c_str=args.phase_channel,
		otsu_thresh=args.otsu_threshold,
		first_open=args.first_opening,
		dist_thresh=args.distance_threshold,
		second_open_size=args.second_opening_size,
		min_area=args.min_cell_area,
		max_area=args.max_cell_area,
		small_merge_thresh=args.small_merge_area_threshold
	)