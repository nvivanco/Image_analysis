import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import shutil
import tifffile
import cv2
from napari_correct_drift import CorrectDrift
from skimage import color
from skimage import filters
from skimage.feature import match_template
from scipy.signal import find_peaks_cwt
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
import HF


def subtract_fov_stack(path_to_mm_channels, FOV, empty_stack_id, ana_peak_ids, method = 'phase', channel_index = 0):
	"""
	For a given FOV, loads the precomputed empty stack and does subtraction on
	all peaks in the FOV designated to be analyzed.

	Args:
		path_to_mm_channels: Path to the directory containing the MM3 channels.
		FOV: str, Field of view to process.
		empty_stack_id: str, ID of the empty stack.
		ana_peak_ids: List of peak IDs (in str type) to analyze.
		method: str, either 'phase' or 'fluor' depending on channel type
		channel_index: integer, index of the phase or fluorophore channel.

	Returns:
		saved subtracted images of mm_channels organized by position and mm_channel
	"""

	#the following first few blocks of code uses inputs to identified directories it will use to run different functions and sets up new directories where files are created into 
	
	path_to_subtracted_channels = os.path.join(path_to_mm_channels, 'subtracted')
	os.makedirs(path_to_subtracted_channels, exist_ok=True)
	path_to_FOV = os.path.join(path_to_subtracted_channels, 'FOV_' + FOV)
	os.makedirs(path_to_FOV, exist_ok=True)

	#following code block creates a dict of channels (numbered) from input and reads the tiffFile and the associated FOV and sorts peak IDs 
	mm3_channels_dict = load_mm_channels(path_to_mm_channels)
	empty_channel_stack = tifffile.imread(mm3_channels_dict[FOV][empty_stack_id])
	ana_peak_ids = sorted(ana_peak_ids)  # Sort for repeatability
	empty_channel_stack_ch = empty_channel_stack[:, channel_index, :, :]

	# Load images for the peak and get phase images
	for peak_id in ana_peak_ids:
		print(peak_id)

		path_to_peak = os.path.join(path_to_FOV, 'region_' + peak_id)
		os.makedirs(path_to_peak, exist_ok=True)

		channel_w_cell_stack = tifffile.imread(mm3_channels_dict[FOV][peak_id])
		channel_w_cell_stack_ch = channel_w_cell_stack[:, channel_index, :, :]

		# Create a list of tuples for multiprocessing
		subtract_pairs = [(empty_channel_stack_ch[i], channel_w_cell_stack_ch[i]) for i in range(len(empty_channel_stack_ch))]

		# Use multiprocessing pool to perform subtraction
		with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
			if method == 'phase':
				subtracted_imgs = pool.map(subtract_phase, subtract_pairs)
			elif method == 'fluor':
				subtracted_imgs = pool.map(subtract_fluor, subtract_pairs)

		subtracted_stack_final = np.stack(subtracted_imgs, axis=0)

		filename = f'subtracted_FOV_{FOV}_region_{peak_id}_c_{str(channel_index)}.tif'
		path = os.path.join(path_to_subtracted_channels, filename)
		tifffile.imwrite(path, subtracted_stack_final)

		for time in range(subtracted_stack_final.shape[0]):
			phase_t_img = subtracted_stack_final[time, :, :]
			time_string = f"{time:0{4}d}"
			filename = f'subtracted_FOV_{FOV}_region_{peak_id}_time_{time_string}_c_{str(channel_index)}.tif'
			path = os.path.join(path_to_peak, filename)
			tifffile.imwrite(path, phase_t_img)

def subtract_phase(params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
	""" Adapted from subtract_phase() of napari-mm3.
	subtract_phase aligns and subtracts an empty phase contrast channel (trap) from a channel containing cells.
	The subtracted image returned is the same size as the image given. It may however include
	data points around the edge that are meaningless but not marked.

	We align the empty channel to the phase channel, then subtract.

	Returns
	channel_subtracted : np.ndarray
		The subtracted image
	"""
	# this is for aligning the empty channel to the cell channel.
	### Pad cropped channel.
	empty_channel_img = params[0]
	channel_with_cells_img = params[1]

	pad_size = 10  # pixel size to use for padding (amount that alignment could be off)
	padded_chnl = np.pad(channel_with_cells_img, pad_size, mode="reflect")

	# ### Align channel to empty using match template.
	# use match template to get a correlation array and find the position of maximum overlap
	match_result = match_template(padded_chnl, empty_channel_img)

	# get row and colum of max correlation value in correlation array
	y, x = np.unravel_index(np.argmax(match_result), match_result.shape)

	# pad the empty channel according to alignment to be overlayed on padded channel.
	empty_paddings = [
		[y, padded_chnl.shape[0] - (y + empty_channel_img.shape[0])],
		[x, padded_chnl.shape[1] - (x + empty_channel_img.shape[1])],
	]

	aligned_empty = np.pad(empty_channel_img, empty_paddings, mode="reflect")
	# now trim it off so it is the same size as the original channel
	aligned_empty = aligned_empty[pad_size: -1 * pad_size, pad_size: -1 * pad_size]

	### Compute the difference between the empty and channel phase contrast images
	# subtract cropped cell image from empty channel.
	channel_subtracted = aligned_empty.astype("int32") - channel_with_cells_img.astype("int32")
	# just zero out anything less than 0.
	channel_subtracted[channel_subtracted < 0] = 0
	channel_subtracted = channel_subtracted.astype("uint16")  # change back to 16bit

	return channel_subtracted


def subtract_fluor(params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
	"""subtract_fluor does a simple subtraction of one image to another. Unlike subtract_phase,
	there is no alignment. Also, the empty channel is subtracted from the full channel.

	Parameters
	image_pair : tuple of length two with; (image, empty_mean)

	Returns
	channel_subtracted : np.array
		The subtracted image.

	Called by
	subtract_fov_stack
	"""
	empty_channel = params[0]
	channel_with_cells = params[1]


	# check frame size of cropped channel and background, always keep crop channel size the same
	crop_size = np.shape(channel_with_cells)[:2]



	empty_size = np.shape(empty_channel)[:2]
	if crop_size != empty_size:
		if crop_size[0] > empty_size[0] or crop_size[1] > empty_size[1]:
			pad_row_length = max(crop_size[0] - empty_size[0], 0)  # prevent negatives
			pad_column_length = max(crop_size[1] - empty_size[1], 0)
			print(pad_column_length)
			empty_channel = np.pad(
				empty_channel,
				[
					[
						np.int(0.5 * pad_row_length),
						pad_row_length - np.int(0.5 * pad_row_length),
					],
					[
						np.int(0.5 * pad_column_length),
						pad_column_length - np.int(0.5 * pad_column_length),
					],
					[0, 0],
				],
				"edge",
			)
		empty_size = np.shape(empty_channel)[:2]
		if crop_size[0] < empty_size[0] or crop_size[1] < empty_size[1]:
			empty_channel = empty_channel[
							: crop_size[0],
							: crop_size[1],
							]

	# subtract empty  channel from fluorophore cell image
	channel_subtracted = channel_with_cells.astype("int32") - empty_channel.astype("int32")

	# just zero out anything less than 0.
	channel_subtracted[channel_subtracted < 0] = 0
	channel_subtracted = channel_subtracted.astype("uint16")  # change back to 16bit

	return channel_subtracted

def load_mm_channels(input_dir):
	"""Group files by FOV and mm_channel id, TIFFs are stacked in tcyx format
	Returns a dictionary in the following format:
	dict[FOV] = {mm_channel_id : '/path/to/tif/file'}
	"""
	file_groups = {}

	for filename in os.listdir(input_dir):
		if filename.endswith('.tif') or filename.endswith('.tiff'):
			match = re.match(r'FOV(\d+)_region_(\d+)\.', filename)
			if match:
				FOV, mm_channel_id = match.groups()

			path = os.path.join(input_dir, filename)
			if FOV not in file_groups:
				file_groups[FOV] = {}
			if mm_channel_id not in file_groups[FOV]:
				file_groups[FOV][mm_channel_id] = path

	return file_groups


def extract_mm_channels(path_to_tcyx_FOVs, chan_w=10, chan_sep=45, crop_wp=10, chan_lp=10, chan_snr=1):
	"""
	Updated 05/15/25: made changes so that when a mask is made it will create a csv 
	of the channels, their ID, and coordinates to be made to track channels further down in pipeline 
	"""
	
	# create an output directory for microfluidic_channels
	path_to_mm_channels = os.path.join(path_to_tcyx_FOVs, 'mm_channels')
	os.makedirs(path_to_mm_channels, exist_ok=True)

	file_group = org_by_timepoint([path_to_tcyx_FOVs])
	font = ImageFont.truetype('/System/Library/Fonts/ArialHB.ttc', 15)
	channels_df_col = ['channel_ID', 'x', 'y', 'cells']

	for position in file_group.keys():
		file_path = file_group[position]['hyperstacked']['stacked']
		FOV_stack_tcyx = tifffile.imread(file_path)
		first_phase_image = FOV_stack_tcyx[0, 0, :, :]
		channels_df = pd.DataFrame(columns = channels_df_col)

		chnl_loc_dict = find_channel_locs(first_phase_image, chan_w, chan_sep, crop_wp, chan_snr)
		image_rows = first_phase_image.shape[0]
		image_cols = first_phase_image.shape[1]
		print("channels identified in FOV " + position)
		consensus_mask, mask_corners_dict = make_consensus_mask(chnl_loc_dict, image_rows,
																image_cols, crop_wp, chan_lp)
		masked_image = first_phase_image * consensus_mask
		# Convert to RGB
		rgb_img = color.gray2rgb(masked_image, channel_axis=-1)
		# Convert 32 to 8 bit for PIL
		scaling_factor = 255 / (np.max(rgb_img) - np.min(rgb_img))
		scaled_img = (rgb_img - np.min(rgb_img)) * scaling_factor
		scaled_img = scaled_img.astype(np.uint8)
		# Convert the masked image to PIL format for text overlay
		pil_image = Image.fromarray(scaled_img)
		draw = ImageDraw.Draw(pil_image)
		fov_text = position
		draw.text((0, 0), text= fov_text, font=font, fill='red')

		for mm_channel in mask_corners_dict.keys():
			ch_text = mm_channel.astype(str)
			x = mask_corners_dict[mm_channel][2]
			y = mask_corners_dict[mm_channel][1]
			channels_df.loc[len(channels_df)] = [int(ch_text), int(x), int(y), 0]
			draw.text((x, y), text=ch_text, font=font, fill='red')
		final_image = np.array(pil_image)
		plt.figure()
		plt.imshow(final_image)
		plt.title('Channels Identified')
		plt.axis('off')  # Hide axis labels
		plt.draw()

		filename = f'FOV{position}_mm_channel_mask.tif'
		path = os.path.join(path_to_mm_channels, filename)
		tifffile.imwrite(path, final_image)

		csv_path = path_to_mm_channels + f'FOV{position}.csv' 
		channels_df.to_csv(csv_path, index=False)
		
		print("saving sliced microfluidic channels as tcyx stacks")
		for trench in mask_corners_dict.keys():
			y1, y2, x1, x2 = mask_corners_dict[trench]
			trench_region = FOV_stack_tcyx[:, :, y1:y2, x1:x2]  # assuming image is stacked as tcyx
			filename = f'FOV{position}_region_{trench}.tif'
			path = os.path.join(path_to_mm_channels, filename)
			tifffile.imwrite(path, trench_region)
	plt.show()

	return path_to_mm_channels


def make_consensus_mask(chnl_loc_dict, image_rows, image_cols, crop_wp=10, chan_lp=10):
	"""
	Generate consensus channel mask for a given fov.
	Adapted from napari-mm3.

	Parameters
	----------
	chnl_loc_dict: dictionary with locations of microfluidic channels
	crop_wp: int channel width padding
	crop_lp: int channel_width padding

	Returns
	-------
	normalized mask of identified mm channels and
	dictionary containing coordinates of each microfluidic channel in an FOV
	"""

	mask_corners_dict = {}

	consensus_mask = np.zeros([image_rows, image_cols])  # mask for labeling entire image
	# for each trench in each image make a single mask
	img_chnl_mask = np.zeros([image_rows, image_cols])

	# and add the channel/peak mask to it
	# Assuming chnl_loc_dict is a NumPy array
	for chnl_peak in chnl_loc_dict:
		peak_ends = chnl_loc_dict[chnl_peak]
		# pull out the peak location and top and bottom location
		# and expand by padding
		x1 = max(chnl_peak - crop_wp, 0)
		x2 = min(chnl_peak + crop_wp, image_cols)
		y1 = max(peak_ends["closed_end_px"] - chan_lp, 0)
		y2 = min(peak_ends["open_end_px"] + chan_lp, image_rows)
		mask_corners_dict[chnl_peak] = [y1, y2, x1, x2]

		# add it to the mask for this image
		img_chnl_mask[y1:y2, x1:x2] = 1

	# add it to the consensus mask
	consensus_mask += img_chnl_mask

	# Normalize consensus mask between 0 and 1.
	consensus_mask = consensus_mask.astype("float32") / float(np.amax(consensus_mask))
	return consensus_mask, mask_corners_dict

def find_channel_locs(image_data, chan_w = 10, chan_sep = 45, crop_wp= 10, chan_snr = 1):
    """
    Adapted from napari-mm3.
    Finds the location of channels from a phase contrast image. The channels are returned in
    a dictionary where the key is the x position of the channel in pixel and the value is a
    dicionary with the open and closed end in pixels in y.

    image data is an np.array of the first phase image of the FOV


    """

    # Detect peaks in the x projection (i.e. find the channels)
    projection_x = image_data.sum(axis=0).astype(np.int32)
    # find_peaks_cwt is a function which attempts to find the peaks in a 1-D array by
    # convolving it with a wave. here the wave is the default Mexican hat wave
    # but the minimum signal to noise ratio is specified
    # *** The range here should be a parameter or changed to a fraction.
    peaks = find_peaks_cwt(
        projection_x, np.arange(chan_w - 5, chan_w + 5), min_snr=chan_snr
    )

    # If the left-most peak position is within half of a channel separation,
    # discard the channel from the list.
    if peaks[0] < (chan_sep / 2):
        peaks = peaks[1:]
    # If the difference between the right-most peak position and the right edge
    # of the image is less than half of a channel separation, discard the channel.
    if image_data.shape[1] - peaks[-1] < (chan_sep / 2):
        peaks = peaks[:-1]

    # Find the average channel ends for the y-projected image
    projection_y = image_data.sum(axis=1)
    # find derivative, must use int32 because it was unsigned 16b before.
    proj_y_d = np.diff(projection_y.astype(np.int32))
    # use the top third to look for closed end, is pixel location of highest deriv
    onethirdpoint_y = int(projection_y.shape[0] / 3.0)
    default_closed_end_px = proj_y_d[:onethirdpoint_y].argmax()
    # use bottom third to look for open end, pixel location of lowest deriv
    twothirdpoint_y = int(projection_y.shape[0] * 2.0 / 3.0)
    default_open_end_px = twothirdpoint_y + proj_y_d[twothirdpoint_y:].argmin()
    default_length = default_open_end_px - default_closed_end_px  # used for checks

    # go through peaks and assign information
    # dict for channel dimensions
    chnl_loc_dict = {}
    # key is peak location, value is dict with {'closed_end_px': px, 'open_end_px': px}

    for peak in peaks:
        # set defaults
        chnl_loc_dict[peak] = {
            "closed_end_px": default_closed_end_px,
            "open_end_px": default_open_end_px,
        }
        # redo the previous y projection finding with just this channel
        channel_slice = image_data[:, peak - crop_wp : peak + crop_wp]
        slice_projection_y = channel_slice.sum(axis=1)
        slice_proj_y_d = np.diff(slice_projection_y.astype(np.int32))
        slice_closed_end_px = slice_proj_y_d[:onethirdpoint_y].argmax()
        slice_open_end_px = twothirdpoint_y + slice_proj_y_d[twothirdpoint_y:].argmin()
        slice_length = slice_open_end_px - slice_closed_end_px

        # check if these values make sense. If so, use them. If not, use default
        # make sure length is not 30 pixels bigger or smaller than default
        # *** This 15 should probably be a parameter or at least changed to a fraction.
        if slice_length + 15 < default_length or slice_length - 15 > default_length:
            continue
        # make sure ends are greater than 15 pixels from image edge
        if slice_closed_end_px < 15 or slice_open_end_px > image_data.shape[0] - 15:
            continue

        # if you made it to this point then update the entry
        chnl_loc_dict[peak] = {
            "closed_end_px": slice_closed_end_px,
            "open_end_px": slice_open_end_px,
        }

    return chnl_loc_dict

def midpoint_distance(line, center):
    # Function to calculate line midpoint distance
    midpoint_x = (line[0][0] + line[0][2]) / 2
    midpoint_y = (line[0][1] + line[0][3]) / 2
    distance = np.sqrt((midpoint_x - center[0])**2 + (midpoint_y - center[1])**2)
    return distance

def crop_around_central_flow(h_lines, w, h, growth_channel_length=400, threshold=1600):
	"""
	Crops an image around the central flow channel based on detected horizontal lines.

	Args:
		h_lines: A list of detected horizontal lines, where each line is represented
					as a NumPy array in the format [[x1, y1, x2, y2]].
		w: Width of the image.
		h: Height of the image.
		growth_channel_length: Desired length of the cropped region along the flow channel.
		threshold: Maximum distance of a line from the image center to be considered.

	Returns:
		A tuple containing the start and end indices for cropping along the vertical axis
		(y-axis) if a suitable line is found, otherwise None.
	"""

	center_y = h // 2
	longest_line = None
	max_length = 0
	
	if h_lines:
		for line in h_lines:
			x1, y1, x2, y2 = line[0]
			length  = np.hypot(x2-x1,y2-y1)
			if length > max_length:
				max_length = length
				longest_line = line

		# Filter lines based on distance from the center and the threshold
		# We'll consider the 'closest_line' found earlier as the primary candidate
		# and then apply the threshold.
		if longest_line is not None:
			y_of_longest_line = longest_line[0][1]
			if abs(y_of_longest_line - center_y) <= threshold:
				# Determine crop boundaries
				crop_start = max(y_of_longest_line, 0)
				crop_end = min(y_of_longest_line + growth_channel_length, h) # Ensure crop_end doesn't exceed image height
				return crop_start, crop_end
			else:
				print(f"The closest line (y={y_of_longest_line}) is outside the {threshold} pixel threshold from the center (y={center_y}).")
				return None
		else:
			print("No suitable line found after median selection.")
			return None
	else:
		print("No horizontal lines were detected in the image.")
		return None

def rotate_stack(path_to_stack, c=0, growth_channel_length=400, closed_ends = 'down'):
	"""Rotates and crops a stack of cyx or tcyx format files.

	Args:
		path_to_stack: Path to the stack of files in string format.
		c: Phase channel index (integer, default=0).
		growth_channel_length: Length in pixels of the growth channel (integer, default=400).
			this really depends on the binning of the image. Images are assummed to be 1x1

	Returns:
		path_to_rotated_images: Path to the directory containing the rotated files (string).

	"""

	# Create output directory for rotated files
	font = ImageFont.truetype('/System/Library/Fonts/ArialHB.ttc', 15)

	path_to_rotated_images = os.path.join(path_to_stack, 'rotated')
	os.makedirs(path_to_rotated_images, exist_ok=True)

	# Group files by timepoint
	file_groups = org_by_timepoint([path_to_stack])

	for position in file_groups.keys():
		print(position)
		file_path = file_groups[position]['hyperstacked']['stacked']
		filename = os.path.basename(file_path)
		stacked_img = tifffile.imread(file_path)
		ref_phase_img = stacked_img[0, c, :, :]  # Assuming phase data is in the first frame

		# Find lines in the reference phase image
		h, w = ref_phase_img.shape
		horizontal_lines, vertical_lines = id_lines(ref_phase_img)
		plot_lines(ref_phase_img, horizontal_lines)

		# Calculate rotation angle
		rotation_angle = calculate_rotation_angle(horizontal_lines)

		# Apply image rotation and generate new image
		ref_rotated_image = apply_image_rotation(ref_phase_img, rotation_angle, closed_ends)

		# Identify lines in the rotated image
		rot_horizontal_lines, rot_vertical_lines = id_lines(ref_rotated_image)
		# find spot to

		# identify coordinates that will be used to crop the image around
		# the central flow using the rotated image as a reference 
		crop_start, crop_end = crop_around_central_flow(rot_horizontal_lines, w, h, growth_channel_length, 2000)

		# Rotate and crop the entire stack, cropping appears to be done only 
		# by inputting the crop start and crop end coordinates for the y input
		rotated_stack = apply_image_rotation(stacked_img, rotation_angle, closed_ends)
		cropped_stack = rotated_stack[:, :, crop_start:crop_end, :]

		# Visualize the cropped stack (optional)
		rgb_img = color.gray2rgb(cropped_stack[0, c, :, :], channel_axis=-1)
		# Convert 32 to 8 bit for PIL
		scaling_factor = 255 / (np.max(rgb_img) - np.min(rgb_img))
		scaled_img = (rgb_img - np.min(rgb_img)) * scaling_factor
		scaled_img = scaled_img.astype(np.uint8)
		# Convert the masked image to PIL format for text overlay
		pil_image = Image.fromarray(scaled_img)
		draw = ImageDraw.Draw(pil_image)
		fov_text = position
		draw.text((0, 0), text= fov_text, font=font, fill='red')
		final_image = np.array(pil_image)
		plt.figure()
		plt.imshow(final_image, cmap='gray')
		plt.axis('off')  # Hide axes labels and ticks
		plt.draw()# Show the plot

		# Save the rotated and cropped stack
		new_filename = f'rotated_{filename}'
		new_path = os.path.join(path_to_rotated_images, new_filename)
		tifffile.imwrite(new_path, cropped_stack)

	plt.show()
	print('Successfully rotated stack')
	return path_to_rotated_images

def detect_clear_image(image):
    laplacian_image = filters.laplace(image)
    blur_score = np.var(laplacian_image)
    if blur_score >= 0:
        return True


def drift_correct(root_dir, experiment_name, pos_list, c=0, ):
	"""
	Arg
	root_dir: parent directory containing multiple 'Pos#' directories,
	each containing tif files of single timepoints and channels of the given position.
	File name is default from the Covert lab microscope.
	experiment_name: unique id to label output files
	c = int representing phase channel index
	output: drift corrected files across multiple positions and timepoints.

	"""

	hyperstacked_path, time_dict = hyperstack_tif_tcyx(root_dir, experiment_name, pos_list , c )

	drift_corrected_path = drift_correction_napari(hyperstacked_path)

	return drift_corrected_path


def hyperstack_tif_tcyx(root_dir, experiment_name, pos_list, c=0):
	"""Renames TIFF files without deleting originals.
	Args:
	input_dir: parent directory.
	experiment_name: The desired experiment name.
	Updated 05/08/25 to ask user if they want the files to be saved in a specific directorty
	not implemented the cleanest so asks for user to select directory with experiment name 
	in order to work properly
    updated 05/15/2025
	pos_list: a list of positions that you want the analysis to work on to save time when not needing to
	analyze every image. format should be 'root_dir/Posx'. If the list is empty it will run through 
	everything as normal	
	"""
	root = Path(root_dir)
	input_dirs = [str(path) for path in root.glob('**//Pos*') if path.is_dir()]

	if pos_list:
		input_dirs = [pos for pos in input_dirs if pos in pos_list]

	
	user_input = input('do you want to save into a specific folder? (Y/N)')
	if user_input.lower() == 'y':
		
		#select which directory (folder) you want to save the output to
		print('select save directory')
		save_dir = HF.select_directory()
		
		#select folder containing experiment name as basename of path (there's probably an easier way to get this)
		print('select experiment name directory')
		experiment_dir = HF.select_directory()

		#the following just creates the appropriate experiment name and direcotries in the folder you want it made 
		experiment_name = os.path.basename(experiment_dir)
		experiment_save_dir = os.path.join(save_dir, experiment_name)
		os.makedirs(experiment_save_dir, exist_ok=True)
		output_dir_path = os.path.join(experiment_save_dir, 'renamed')
		hyperstacked_path = os.path.join(experiment_save_dir, 'hyperstacked')
	else:
		output_dir_path = os.path.join(root_dir, 'renamed')
		hyperstacked_path = os.path.join(root_dir, 'hyperstacked')

	# Create output directory if it doesn't exist

	os.makedirs(output_dir_path, exist_ok=True)
	os.makedirs(hyperstacked_path, exist_ok=True)

	time_clear_dict = {}

	file_groups = org_by_timepoint(input_dirs)
	for position, time in sorted(file_groups.items()):
		time_clear_dict[position] = {}
		time_stacked_image_data = []
		for time, channels in sorted(time.items()):
			image_data = []
			for channel, image_path in sorted(channels.items()):
				new_filename = f'{experiment_name}_t{time:04.0f}xy{position}c{channel}.tif'
				new_path = os.path.join(output_dir_path, new_filename)
				try:
					# Copy the file to the new path
					shutil.copy(str(image_path), str(new_path))
					channel_image = tifffile.imread(new_path)
					image_data.append(channel_image)

				except OSError as e:
					print(f'Error copying file: {e}')
			stacked_image = np.stack(image_data, axis=0)  # Assuming channels are the first dimension
			phase_image = stacked_image[c, :, :]
			if detect_clear_image(phase_image):  # only time stack clear images
				time_stacked_image_data.append(stacked_image)
				time_clear_dict[position][time] = 'clear'
			else:
				time_clear_dict[position][time] = 'blurry'
				print('blurry')
		hyperstacked_image = np.stack(time_stacked_image_data, axis=0)  # time as the first dimension
		output_hyperstacked_file = Path(hyperstacked_path) / f"{experiment_name}_xy{position}.tif"
		tifffile.imwrite(str(output_hyperstacked_file), hyperstacked_image)

	return hyperstacked_path, time_clear_dict


def drift_correction_napari(hyperstacked_path):
	output_dir_path = os.path.join(hyperstacked_path, 'drift_corrected')
	os.makedirs(output_dir_path, exist_ok=True)

	for filename in os.listdir(hyperstacked_path):
		if filename.endswith('.tif') or filename.endswith('.tiff'):
			if re.match(r'(.*)_xy(\d+)\.', filename):
				print(filename)
				match = re.match(r'(.*)_xy(\d+)\.', filename)
				experiment, position = match.groups()
				img_path = os.path.join(hyperstacked_path, filename)
				hyperstacked_img = tifffile.imread(img_path)
				# multi-channel 2D-movie
				cd = CorrectDrift(dims="tcyx", data=hyperstacked_img)
				# estimate drift table
				drifts = cd.estimate_drift(t0=0, channel=0, increment=1, upsample_factor=15, mode='relative')
				# correct drift
				img_cor = cd.apply_drifts(drifts)
				img_cor_file = Path(output_dir_path) / f"drift_cor_{experiment}_xy{position}.tif"
				tifffile.imwrite(str(img_cor_file), img_cor)
	return output_dir_path


def org_by_timepoint(input_dirs):
	"""Group files by time and channel id, it does not take into account the z axis
	Reads in files in the format exported by the Covert lab scope,
	which is as follows: 'img_channel(\d+)_position(\d+)_time(\d+)_z(\d+)\.'

	Returns a dictionary in the following format:
	dict[time_frame] = {channel_id : '/path/to/tif/file'}
	"""

	time = 'hyperstacked'
	channel = 'stacked'
	position = '0'

	file_groups = {}

	for input_dir in input_dirs:
		for filename in os.listdir(input_dir):
			if filename.endswith('.tif') or filename.endswith('.tiff'):
				match = re.match(r'img_channel(\d+)_position(\d+)_time(\d+)_z(\d+)\.', filename)
				if match:
					channel, position, time, z = match.groups()
					time = int(time)
				elif re.match(r'(.*)_t(\d+)xy(\d+)\.', filename):
					match = re.match(r'(.*)_t(\d+)xy(\d+)\.', filename)
					experiment, time, position = match.groups()
					time = int(time)
				else:
					match = re.match(r'(.*)_xy(\d+)\.', filename)
					if match:
						experiment, position = match.groups()
				path = os.path.join(input_dir, filename)
				if position not in file_groups:
					file_groups[position] = {}
				if time not in file_groups[position]:
					file_groups[position][time] = {}
				if channel not in file_groups[position][time]:
					file_groups[position][time][channel] = path

	return file_groups


def unstack_tcyx_to_cyx(path_to_hyperstacked):
	"""
	input_dir: directory where movies are hyperstacked as tcyx
	output_dir: The output directory for TIFF files stacked as cyx
	"""

	# Create output directory if it doesn't exist
	output_dir_path = os.path.join(path_to_hyperstacked, 'unstacked_files')
	os.makedirs(output_dir_path, exist_ok=True)

	file_groups = org_by_timepoint([path_to_hyperstacked])
	for position, time in sorted(file_groups.items()):
		for time, channels in sorted(time.items()):
			for channel, image_path in sorted(channels.items()):
				filename = os.path.basename(image_path)
				match = re.match(r'(.*)_xy(\d+)\.', filename)
				if match:
					experiment, position = match.groups()
					xy_dir = os.path.join(output_dir_path, position)
					os.makedirs(xy_dir, exist_ok=True)
					hyperstacked_img = tifffile.imread(image_path)
					for time_index in range(hyperstacked_img.shape[0]):
						cyx_image = hyperstacked_img[time_index, :, :, :]
						for channel_index in range(cyx_image.shape[0]):
							yx_image = cyx_image[channel_index, :, :]
							output_yx_file = Path(xy_dir) / f"{experiment}_t{time_index:04.0f}xy{position}_c{channel_index:04.0f}.tif"
							tifffile.imwrite(str(output_yx_file), yx_image)

#used to calculate (keep positive always)
def calculate_line_angle(x1, y1, x2, y2):
	dx = x2 - x1
	dy = y2 - y1
	angle = np.arctan2(dy, dx) * 180 / np.pi
	return angle

# this is the function that finds all lines using a HoughLines transform
# it takes in an image transforms it into 8-bit and returns a list of lines
# each with a start and endpoint coordinate [x1,y1.x2,y2]
def find_lines(img):
	normalized_img = (img / img.max() * 255).astype(np.uint8)
	edges = cv2.Canny(normalized_img, 50, 150)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=130, maxLineGap=100) #max line gap is going to depend on pixel binning. currently set for 1x1 bin
	return lines

# this function finds all the lines in the image using find_lines
# then it filters out those lines into vertical and horizontal lines 
# filter is done by checking the angle of those lines 
def id_lines(img):
	lines = find_lines(img)
	h_lines = []
	v_lines = []
	count = 0
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			angle = calculate_line_angle(x1, y1, x2, y2)
			if abs(angle) < 30:  # Adjust threshold as needed
				h_lines.append(line)
				count += 1
			elif 60 <= abs(angle) <= 120:  # Adjust threshold as needed
				v_lines.append(line)
				count += 1
			if count >= 50:
				break
	return h_lines, v_lines

#this is the function that calculates the angle to be rotated. 
#this is the one that should be changed to affect rotation
# it calculates the angles of every line then finds the average angle 

def calculate_rotation_angle(lines):
	"""calculate rotation angle based on phase image"""
	angles = []
	for line in lines:
		# Calculate angle of the line
		x1, y1, x2, y2 = line[0]
		angle = calculate_line_angle(x1, y1, x2, y2)
		angles.append(abs(angle))
	average_angle = sum(angles) / len(angles)
	return -average_angle

#this function plots a list of lines over an image 
def plot_lines(original_img, lines):
	plt.figure()
	plt.imshow(original_img, cmap='gray')
	if lines:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			plt.plot([x1, x2], [y1, y2], color='green', linewidth=2)
		plt.axis('off')
		plt.draw()
	plt.show()


def apply_image_rotation(image_stack, rotation_angle, closed_ends = 'down'):
	"""Applies rotation to an image stacked as tcyx.

	Args:
		image: image in Grey or BGR format for OpenCV
		rotation_angle: The rotation angle in degrees.
		closed_ends: orientation of closed ends of growth channels,
		can be "up" or "down"

	Returns:
		Rotated image in BGR format.
	"""
	rotated_stack = np.zeros_like(image_stack)
	h = None
	w = None
	adjusting_angle = 0
	if closed_ends == 'up':
		adjusting_angle = 180

	# assumes tcyx format
	if image_stack.ndim == 4:
		h, w = image_stack.shape[2:]
		#define center by which images will be rotated 
		center = (w // 2, h // 2)
		print('Rotation angle:')
		print(rotation_angle)
		M = cv2.getRotationMatrix2D(center, rotation_angle-adjusting_angle, 1.0)
		for time in range(image_stack.shape[0]):
			for channel in range(image_stack.shape[1]):
				rotated_stack[time, channel] = cv2.warpAffine(image_stack[time, channel], M, (w, h))

	elif image_stack.ndim == 3:
		h, w = image_stack.shape[1:]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, rotation_angle-adjusting_angle, 1.0)
		for channel in range(image_stack.shape[0]):
			rotated_stack[channel] = cv2.warpAffine(image_stack[channel], M, (w, h))

	elif image_stack.ndim == 2:
		h, w = image_stack.shape
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, rotation_angle-adjusting_angle, 1.0)
		rotated_stack = cv2.warpAffine(image_stack, M, (w, h))

	return rotated_stack