import argparse
from mmtrack import pre_process_mm


def run_background_subtraction(input_path, fov, empty_id, ana_ids, phase_idx, fluor_idx):
	"""
	Performs background subtraction on extracted Mother Machine channels using
	user-specified trench IDs and channel indices.

	Args:
		input_path (str): Path to the directory containing the extracted channels.
		fov (str): The Field of View ID (e.g., '007').
		empty_id (str): The ID of the empty stack/trench used for background.
		ana_ids (list): List of trench IDs to be analyzed.
		phase_idx (int): The index of the phase contrast channel.
		fluor_idx (int): The index of the fluorescent channel.
	"""
	print(f"Starting background subtraction for files in: {input_path}")
	print(f"Using FOV: {fov}, Empty Stack ID: {empty_id}, Analysis IDs: {ana_ids}")
	print(f"Phase Channel Index: {phase_idx}, Fluorescent Channel Index: {fluor_idx}")

	# --- PHASE CONTRAST BACKGROUND SUBTRACTION ---
	print("\n--- 1. Subtracting Phase Contrast Background ---")
	pre_process_mm.subtract_fov_stack(
		input_path,
		fov,
		empty_id,
		ana_ids,
		method='phase',
		channel_index=phase_idx  # Now uses the command-line argument
	)

	# --- FLUORESCENCE BACKGROUND SUBTRACTION ---
	print("\n--- 2. Subtracting Fluorescent Background ---")
	pre_process_mm.subtract_fov_stack(
		input_path,
		fov,
		empty_id,
		ana_ids,
		method='fluor',
		channel_index=fluor_idx  # Now uses the command-line argument
	)

	print("\n✅ Pipeline Stage 2 (Background Subtraction) Complete.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Stage 2: Background subtraction using a user-specified empty trench ID."
	)

	# --- Mandatory I/O and User-Inspected Arguments ---
	parser.add_argument(
		'--input-path',
		required=True,
		type=str,
		help="Path to the directory containing the extracted channels (output of Stage 1)."
	)
	parser.add_argument(
		'--fov',
		required=True,
		type=str,
		help="The Field of View ID (e.g., '007')."
	)
	parser.add_argument(
		'--empty-stack-id',
		required=True,
		type=str,
		help="The ID of the empty trench/stack to use for background subtraction (e.g., '765')."
	)
	parser.add_argument(
		'--ana-peak-ids',
		required=True,
		nargs='+',
		help="A space-separated list of trench IDs to analyze (e.g., 992 1219 1749)."
	)

	# --- Optional Channel Index Parameters (New) ---
	parser.add_argument(
		'--phase-index',
		type=int,
		default=0,
		help="The channel index used for Phase Contrast images. Default: 0."
	)
	parser.add_argument(
		'--fluor-index',
		type=int,
		default=1,
		help="The channel index used for Fluorescent images. Default: 1."
	)

	args = parser.parse_args()

	run_background_subtraction(
		input_path=args.input_path,
		fov=args.fov,
		empty_id=args.empty_stack_id,
		ana_ids=args.ana_peak_ids,
		phase_idx=args.phase_index,
		fluor_idx=args.fluor_index
	)