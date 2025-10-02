import argparse
import os
import json
import pandas as pd
import numpy as np
import tifffile
import glob
from skimage.measure import regionprops_table, label
from mmtrack import plot_cells


def get_tiff_frame_count(file_path):
	"""
	Reads a TIFF file using imread and returns the index of the last time frame (T - 1).
	NOTE: This loads the entire file into memory.
	"""
	try:
		# Load the entire image stack into memory
		img_stack = tifffile.imread(file_path)

		shape = img_stack.shape

		# Assume the time axis (T) is the first dimension
		if len(shape) >= 3:
			# The last index is T - 1
			return shape[0] - 1
		else:
			# Single 2D image
			return 0

	except FileNotFoundError:
		print(f"Error: TIFF file not found at {file_path}")
		return 0
	except Exception as e:
		print(f"Error reading TIFF file {file_path}: {e}")
		return 0

def run_feature_extraction(base_dir, time_range_json, phase_c_str, fluor_c_str):
	"""
	Creates kymographs and extracts cell properties.
	Uses provided time ranges or defaults to the full TIFF stack size.
	"""

	# 1. Prepare Experiment List and Time Ranges
	if time_range_json:
		try:
			time_range_dict = json.loads(time_range_json)
		except json.JSONDecodeError:
			print("ERROR: Could not parse the time range dictionary. Ensure it is valid JSON.")
			return
	else:
		# If no JSON is provided, the script will automatically discover files
		time_range_dict = {}

	if not time_range_dict:
		print("Time range dictionary not provided. Discovering all segmented files for full range tracking...")

		# --- File Discovery and Auto-Population (CRUCIAL STEP) ---
		# Search for all segmented files across all subdirectories of base_dir
		# Assumes segmented files are deep inside the experiment folders:
		# {base_dir}/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/mm3_segmented_...

		# Use a flexible glob pattern to find segmented files
		segmented_files = glob.glob(os.path.join(
			base_dir,
			'*/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/mm3_segmented*.tif'
		))

		if not segmented_files:
			print("ERROR: Could not find any segmented TIFF files matching the pattern. Check 'base-dir'.")
			return

		for f_path in segmented_files:
			# Reverse-engineer metadata from the file path
			# Example path: .../EXP_FOLDER/.../mm3_segmented_subtracted_FOV_007_region_992_c_0.tif

			# 1. Get components after the 'subtracted' folder
			relative_path = f_path.split('subtracted')[-1].strip(os.sep)

			# 2. Extract folder (experiment name)
			exp_folder = f_path.split('hyperstacked')[0].split(base_dir + os.sep)[-1].strip(os.sep)

			# 3. Extract FOV and Peak ID from filename (assuming fixed naming convention)
			# Filename should look like: mm3_segmented_subtracted_FOV_007_region_992_c_0.tif
			parts = os.path.basename(f_path).split('_')

			try:
				fov_id_idx = parts.index('FOV') + 1
				peak_id_idx = parts.index('region') + 1
				fov_id = parts[fov_id_idx]
				peak_id = parts[peak_id_idx]
			except (ValueError, IndexError):
				print(f"WARNING: Could not parse metadata from filename: {os.path.basename(f_path)}. Skipping.")
				continue

			# Auto-populate the dict structure
			if exp_folder not in time_range_dict:
				time_range_dict[exp_folder] = {}
			if fov_id not in time_range_dict[exp_folder]:
				time_range_dict[exp_folder][fov_id] = {}

			# Placeholder for dynamic calculation inside the loop
			time_range_dict[exp_folder][fov_id][peak_id] = {'start': 0, 'end': None}

	# 2. Start Processing Loop
	print(f"Starting feature extraction across {len(time_range_dict)} experiments.")

	for folder, fov_dict in time_range_dict.items():
		print(f"\nProcessing Experiment: {folder}")
		all_cells_pd = pd.DataFrame()

		for fov_id in fov_dict.keys():
			trench_time_ranges = fov_dict[fov_id]
			print(f"  FOV: {fov_id}, Trenches: {list(trench_time_ranges.keys())}")

			for peak_id, time_info in trench_time_ranges.items():

				# --- Path Construction ---
				base_file_path = os.path.join(base_dir, folder, 'hyperstacked', 'drift_corrected', 'rotated',
											  'mm_channels', 'subtracted')
				path_to_phase_stack = os.path.join(base_file_path,
												   f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif')
				path_to_labeled_stack = os.path.join(base_file_path,
													 f'mm3_segmented_subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif')
				path_to_fluor_stack = os.path.join(base_file_path,
												   f'subtracted_FOV_{fov_id}_region_{peak_id}_c_{fluor_c_str}.tif')

				# --- Dynamic Time Range Assignment ---
				start = time_info['start']
				end = time_info.get('end')

				if end is None:
					# Calculate end frame dynamically using the Phase stack file
					end = get_tiff_frame_count(path_to_phase_stack)
					if end == 0:
						print(f"    WARNING: Could not determine frame count for {peak_id}. Skipping.")
						continue

				# 2. Read Image Stacks
				try:
					phase_stack = tifffile.imread(path_to_phase_stack)
					labeled_stack = tifffile.imread(path_to_labeled_stack)
					fluor_stack = tifffile.imread(path_to_fluor_stack)
				except FileNotFoundError as e:
					print(f"    WARNING: Required file not found for {peak_id}: {e}. Skipping.")
					continue

				# 3. Create Kymographs (Phase, Mask, Fluor)
				print(f"    -> Creating kymographs for Trench {peak_id} (Frames {start}-{end})...")
				output_base_dir = os.path.join(base_dir, folder, 'hyperstacked', 'drift_corrected', 'rotated',
											   'mm_channels', 'subtracted')
				mask_output_dir = os.path.join(output_base_dir, 'mask_kymos')
				fluor_output_dir = os.path.join(output_base_dir, 'fluor_kymos')
				os.makedirs(mask_output_dir, exist_ok=True)
				os.makedirs(fluor_output_dir, exist_ok=True)

				# Reusing existing output directory variables
				phase_kymograph = plot_cells.create_kymograph(phase_stack, start, end, fov_id, peak_id, output_base_dir)
				mask_kymograph = plot_cells.create_kymograph(labeled_stack, start, end, fov_id, peak_id,
															 mask_output_dir)
				fluor_kymograph = plot_cells.create_kymograph(fluor_stack, start, end, fov_id, peak_id,
															  fluor_output_dir)

				# 4. Label and Extract Properties
				labeled_kymograph_mask = label(mask_kymograph)

				# Phase Properties
				props = regionprops_table(
					labeled_kymograph_mask, phase_kymograph,
					properties=['label', 'area', 'coords', 'centroid', 'axis_major_length',
								'axis_minor_length', 'intensity_mean', 'intensity_max', 'intensity_min']
				)

				# Fluor Properties
				props_fluor = regionprops_table(
					labeled_kymograph_mask, fluor_kymograph,
					properties=['label', 'intensity_mean', 'intensity_max', 'intensity_min']
				)

				# 5. Merge and Process DataFrames
				region_phase_df = pd.DataFrame(props)
				region_fluor_df = pd.DataFrame(props_fluor)

				full_region_df = region_phase_df.merge(region_fluor_df, how='inner', on='label',
													   suffixes=('_phase', '_fluor'))

				labeled_stack_px_width = labeled_stack.shape[-1]
				mask_kymograph_px_width = mask_kymograph.shape[-1]

				plot_cells.add_time_frame_df(
					full_region_df, labeled_stack_px_width, mask_kymograph_px_width,
					x_centroid_col='centroid-1'
				)

				# 6. Add Metadata and Concatenate
				full_region_df['experiment_name'] = folder
				full_region_df['FOV'] = fov_id
				full_region_df['trench_id'] = peak_id

				if all_cells_pd.empty:
					all_cells_pd = full_region_df
				else:
					all_cells_pd = pd.concat([all_cells_pd, full_region_df], ignore_index=True)

		# 7. Save the consolidated DataFrame for the experiment
		all_cells_filename = os.path.join(base_dir, f'all_cell_data_{folder}.pkl')
		all_cells_pd.to_pickle(all_cells_filename)
		print(f"  -> Saved all cell features for {folder} to {all_cells_filename}")

	print("\n Pipeline Stage 4 (Feature Extraction) Complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Stage 4: Create kymographs and extract cell morphological/intensity properties into a DataFrame."
	)

	# --- Mandatory I/O Argument ---
	parser.add_argument(
		'--base-dir',
		required=True,
		type=str,
		help="Base path containing all experiment folders (e.g., /path/to/DuMM_image_analysis)."
	)
	# --- Time Range Dictionary is now OPTIONAL ---
	parser.add_argument(
		'--time-range-dict',
		required=False,  # Changed to False
		type=str,
		default='',  # Changed default to empty string
		help='JSON string defining specific start/end time frames for tracking. If omitted, the full TIFF range is used for all detected segmented files. Format: {"exp": {"FOV": {"trench": {"start": 0, "end": 75}}}}.'
	)

	# --- Optional Channel Index Arguments ---
	parser.add_argument(
		'--phase-channel', type=str, default='0', help="Phase channel index as string. Default: '0'."
	)
	parser.add_argument(
		'--fluor-channel', type=str, default='1', help="Fluor channel index as string. Default: '1'."
	)

	args = parser.parse_args()

	run_feature_extraction(
		base_dir=args.base_dir,
		time_range_json=args.time_range_dict,
		phase_c_str=args.phase_channel,
		fluor_c_str=args.fluor_channel
	)