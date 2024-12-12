import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from skimage import segmentation, morphology
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

def segment_chnl_stack(path_to_phase_channel_stack,
					   output_path,
					   OTSU_threshold=1,
					   distance_threshold=1,
					   second_opening_size=1,
					   min_object_size=25):
	"""
	For a given fov and peak (channel), do segmentation for all images in the
	subtracted .tif stack.
	Adapted from napari-mm3
	"""

	path_to_mm3_seg = os.path.join(output_path, 'mm3_segmentation')
	os.makedirs(path_to_mm3_seg, exist_ok=True)

	save_to_path = os.path.dirname(path_to_phase_channel_stack)
	filename = os.path.basename(path_to_phase_channel_stack)
	phase_stack = tifffile.imread(path_to_phase_channel_stack)


	# image by image for debug
	segmented_imgs = []
	for time in range(phase_stack.shape[0]):
		unstacked_seg_image = segment_image(phase_stack[time, :, :],
											OTSU_threshold,
											distance_threshold,
											second_opening_size,
											min_object_size)

		unstacked_seg_image = unstacked_seg_image.astype("uint8")
		unstacked_seg_filename = f'mask{time:03d}.tiff'
		unstacked_path = os.path.join(path_to_mm3_seg, unstacked_seg_filename)
		tifffile.imwrite(unstacked_path, unstacked_seg_image)

		segmented_imgs.append(unstacked_seg_image)

	# stack them up along a time axis
	segmented_imgs = np.stack(segmented_imgs, axis=0)


	seg_filename = f'mm3_segmented_{filename}'
	path = os.path.join(save_to_path, seg_filename)
	tifffile.imwrite(path, segmented_imgs)


def segment_image(image,
                  OTSU_threshold=1,
                  distance_threshold=1,
                  second_opening_size=1,
                  min_object_size=25):
    """
    Adapted From OTSU segementation on napari-mm3, morph uses diagonal footprint
    Segments a subtracted image and returns a labeled image

    Parameters
    ----------
    image : a ndarray which is an image. This should be the subtracted image

    Returns
    -------
    labeled_image : a ndarray which is also an image. Labeled values, which
        should correspond to cells, all have the same integer value starting with 1.
        Non labeled area should have value zero.
    """

    # threshold image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresh = threshold_otsu(image)  # finds optimal OTSU threshhold value
    except:
        return np.zeros_like(image)

    threshholded = image > OTSU_threshold * thresh  # will create binary image

    # Opening = erosion then dialation.
    # opening smooths images, breaks isthmuses, and eliminates protrusions.
    # "opens" dark gaps between bright features.
    # Create a diagonal line-shaped footprint
    diagonal_footprint = np.zeros((3, 3))
    np.fill_diagonal(diagonal_footprint, 1)

    morph = morphology.binary_opening(threshholded, diagonal_footprint)

    # if this image is empty at this point (likely if there were no cells), just return
    # zero array
    if np.amax(morph) == 0:
        return np.zeros_like(image)

    ### Calculate distance matrix, use as markers for random walker (diffusion watershed)
    # Generate the markers based on distance to the background
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distance = ndi.distance_transform_edt(morph)

    # threshold distance image
    distance_thresh = np.zeros_like(distance)
    distance_thresh[distance < distance_threshold] = 0
    distance_thresh[distance >= distance_threshold] = 1

    # do an extra opening on the distance
    distance_opened = morphology.binary_opening(
        distance_thresh, morphology.disk(second_opening_size)
    )

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(distance_opened)
    # remove small objects. Remove small objects wants a
    # labeled image and will fail if there is only one label. Return zero image in that case
    # could have used try/except but remove_small_objects loves to issue warnings.
    labeled, label_num = morphology.label(cleared, connectivity=1, return_num=True)
    if label_num > 1:
        labeled = morphology.remove_small_objects(labeled, min_size=min_object_size)
    else:
        # if there are no labels, then just return the cleared image as it is zero
        return np.zeros_like(image)

    # relabel now that small objects and labels on edges have been cleared
    markers = morphology.label(labeled, connectivity=1)

    # just break if there is no label
    if np.amax(markers) == 0:
        return np.zeros_like(image)

    # the binary image for the watershed, which uses the unmodified OTSU threshold
    threshholded_watershed = threshholded
    # threshholded_watershed = segmentation.clear_border(threshholded_watershed)

    # label using the random walker (diffusion watershed) algorithm
    try:
        # set anything outside of OTSU threshold to -1 so it will not be labeled
        markers[threshholded_watershed == 0] = -1
        # here is the main algorithm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_image = segmentation.random_walker(-1 * image, markers)
        # put negative values back to zero for proper image
        labeled_image[labeled_image == -1] = 0
    except:
        return np.zeros_like(image)

    return labeled_image
def display_segmentation(phase_path, mask_path, start=0, end=20):
	if os.path.isfile(phase_path):
		phase_stack = tifffile.imread(phase_path)
	elif os.path.isdir(phase_path):
		phase_files = read_phase_sub(phase_path)
		phase_stack = stack_images_tyx(phase_files)
	if os.path.isfile(mask_path):
		mask_stack = tifffile.imread(mask_path)
	elif os.path.isdir(mask_path):
		mask_files = read_celltk_masks(mask_path)
		mask_stack = stack_images_tyx(mask_files)

	final_mask = mask_stack[end, :, :]
	total_number_labels = len(np.unique(final_mask))

	number_of_colors_needed = max(20, total_number_labels)
	# Get a colormap with  distinct colors, for potential cell labels
	cmap = plt.cm.get_cmap('tab20', number_of_colors_needed)
	# Create a dictionary mapping integers to hex color codes
	color_dict = {}
	for j in range(number_of_colors_needed + 1):
		color = cmap(j)[:3]  # Extract RGB values
		color_dict[j] = color

	num_images = len(range(start, end, 1))  # need to add a way to calculate images if numbers are not consecutive
	num_cols = num_images
	num_rows = 1
	fig_width = num_cols / num_rows

	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, phase_stack.shape[2]),
							gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
	# Flatten the axs array for easier indexing
	axs = axs.flatten()

	for i in range(start, end, 1):
		phase = phase_stack[i, :, :]
		mask = mask_stack[i, :, :]

		# Create a colormapped image
		colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
		for x in range(mask.shape[0]):
			for y in range(mask.shape[1]):
				colored_mask[x, y] = color_dict[mask[x, y]]

		# Access the specific subplot and plot the image
		axs[i - start].imshow(phase, cmap='gray')
		axs[i - start].imshow(colored_mask, alpha=0.4)
		axs[i - start].axis('off')

	patches = []
	for label, color in color_dict.items():
		patches.append(mpatches.Patch(color=color, label=f"Label {label}"))
	plt.legend(handles=patches, bbox_to_anchor=(1.05, num_rows), loc='upper left')
	plt.show()

def read_celltk_masks(path):
    file_groups = {}
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            match = re.match(r'mask(\d+)\.', filename)
            if match:
                time = match.groups()[0]
                time = int(time)
                file_path = os.path.join(path, filename)
                if time not in file_groups:
                    file_groups[time] = file_path
    return file_groups

def read_phase_sub(path):
    file_groups = {}
    for filename in os.listdir(path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            match = re.match(r'phase_subtracted_region_(\d+)_time_(\d+)\.', filename)
            if match:
                time = match.groups()[1]
                time = int(time)
                file_path = os.path.join(path, filename)
                if time not in file_groups:
                    file_groups[time] = file_path
    return file_groups


def stack_images_tyx(file_groups):
    image_data = []
    for time, image_path in sorted(file_groups.items()):
        image = tifffile.imread(image_path)
        image_data.append(image)
    stacked_image = np.stack(image_data, axis=0)  # Assuming time is the first dimension
    return stacked_image


