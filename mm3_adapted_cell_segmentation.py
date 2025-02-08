import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from skimage import segmentation, morphology, measure
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import networkx as nx

def segment_chnl_stack(path_to_phase_channel_stack,
					   output_path,
					   OTSU_threshold=1.5,
					   first_opening=5,
					   distance_threshold=3,
					   second_opening_size=1,
					   min_object_size=5,
					   min_cell_area=200,
					   max_cell_area=700,
					   small_merge_area_threshold=50):
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
											first_opening,
											distance_threshold,
											second_opening_size,
											min_object_size,
											min_cell_area,
											max_cell_area,
											small_merge_area_threshold)

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
                  OTSU_threshold=1.5,
                  first_opening=5,
                  distance_threshold=3,
                  second_opening_size=1,
                  min_object_size=5,
                  min_cell_area=200,
                  max_cell_area=700,
                  small_merge_area_threshold=50):
    """Segments an image with size filtering and improved structure."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresh = threshold_otsu(image)
    except:
        return np.zeros_like(image)

    thresholded = image > OTSU_threshold * thresh

    morph = morphology.binary_opening(thresholded, morphology.disk(first_opening))

    if np.amax(morph) == 0:
        return np.zeros_like(image)

    distance = ndi.distance_transform_edt(morph)

    distance_thresh = distance >= distance_threshold

    distance_opened = morphology.binary_opening(distance_thresh, morphology.disk(second_opening_size))

    cleared = segmentation.clear_border(distance_opened)

    labeled, num_labels = morphology.label(cleared, connectivity=1, return_num=True)

    if num_labels == 0:
        return np.zeros_like(image)

    labeled = morphology.remove_small_objects(labeled, min_size=min_object_size)

    markers = morphology.label(labeled, connectivity=1)

    if np.amax(markers) == 0:
        return np.zeros_like(image)

    thresholded_watershed = thresholded

    try:
        markers[thresholded_watershed == 0] = -1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_image = segmentation.random_walker(-1 * image, markers)
        labeled_image[labeled_image == -1] = 0
    except:
        return np.zeros_like(image)

    # Bounding Box Merging
    regions = measure.regionprops(labeled_image)
    if regions:
        bboxes = np.array([region.bbox for region in regions])
        areas = np.array([region.area for region in regions])
        graph = nx.Graph()
        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes):
                if i != j and overlap(bbox1, bbox2) and areas[i] < small_merge_area_threshold and areas[j] < small_merge_area_threshold:
                    print('merge regions')
                    graph.add_edge(regions[i].label, regions[j].label)

        # Find connected components (merged regions)
        for component in nx.connected_components(graph):
            if len(component) > 1:  # Only merge if there's more than one region
                merged_mask = np.isin(labeled_image, list(component))
                new_label = np.max(labeled_image) + 1
                labeled_image[merged_mask] = new_label

                for label in component:
                    labeled_image[labeled_image == label] = 0

        # Relabel after merging
        labeled_image, num_labels = morphology.label(labeled_image, connectivity=1, return_num=True)


    # Size Filtering
    if min_cell_area is not None or max_cell_area is not None:
        filtered_labeled_image = np.zeros_like(labeled_image)
        regions = measure.regionprops(labeled_image)
        for region in regions:
            area = region.area
            if (min_cell_area is None or area >= min_cell_area) and (max_cell_area is None or area <= max_cell_area):
                filtered_labeled_image[region.coords[:, 0], region.coords[:, 1]] = region.label

        labeled_image = filtered_labeled_image

    return labeled_image

def overlap(bbox1, bbox2):
    """Checks if two bounding boxes overlap."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    return x1 < x2 and y1 < y2

def display_segmentation(path_to_original_stack, fov_id, peak_id, mask_path=None, alpha=0.5, cells_df=None, start=0, end=20):
	# Check if files exist
	# plotting lineage trees for complete cells
	general_dir = os.path.dirname(path_to_original_stack)
	plot_out_dir = os.path.join(general_dir, 'kymographs')
	os.makedirs(plot_out_dir, exist_ok=True)
	if os.path.isfile(path_to_original_stack):
		phase_stack = tifffile.imread(path_to_original_stack)

		num_images = len(range(start, end, 1))  # need to add a way to calculate images if numbers are not consecutive

		# Calculate figure size
		figxsize = phase_stack.shape[2] * num_images / 100.0
		figysize = phase_stack.shape[1] / 100.0

		fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(figxsize, figysize), facecolor="white",
								edgecolor="black",
								gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'top': 1, 'bottom': 0})
		# Flatten the axs array for easier indexing
		axs = axs.flatten()
		color_dict = {}
		kymographs_gray = []

		if mask_path is None and cells_df is None:
			for i in range(start, end, 1):
				phase = phase_stack[i, :, :]
				axs[i - start].imshow(phase, cmap='gray')
				axs[i - start].set_yticks([])
				axs[i - start].set_xticks([])
				# axs[i - start].set_xlabel(f"{i}", fontsize=8)
				if phase.ndim == 3:  # Check if it is an RGB image
					kymograph_gray = np.mean(phase, axis=2)  # Average across color channels
				else:
					kymograph_gray = phase  # If it's already grayscale, just copy
				kymographs_gray.append(kymograph_gray)
			combined_kymograph = np.concatenate(kymographs_gray, axis=1)  # Concatenate horizontally
			lin_filename = f'{fov_id}_{peak_id}.tif'
			lin_filepath = os.path.join(plot_out_dir, lin_filename)
			tifffile.imwrite(lin_filepath, combined_kymograph)


		elif mask_path is not None:
			if os.path.isfile(mask_path):
				mask_stack = tifffile.imread(mask_path)
				final_mask = mask_stack[end, :, :]
				total_number_labels = len(np.unique(final_mask))

				number_of_colors_needed = max(20, total_number_labels)
				# Get a colormap with  distinct colors, for potential cell labels
				cmap = plt.cm.get_cmap('tab20', number_of_colors_needed)
				# Create a dictionary mapping integers to hex color codes
				for j in range(number_of_colors_needed + 1):
					color = cmap(j)[:3]  # Extract RGB values
					color_dict[j] = color

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
					axs[i - start].imshow(colored_mask, alpha=alpha)
					axs[i - start].set_yticks([])
					axs[i - start].set_xticks([])
					axs[i - start].set_xlabel(f"{i}", fontsize=8)

		elif cells_df is not None:
			unique_cell_ids = cells_df['cell_id'].unique()
			total_number_labels = len(unique_cell_ids)
			number_of_colors_needed = max(20, total_number_labels)
			# Get a colormap with  distinct colors, for potential cell labels
			cmap = plt.cm.get_cmap('tab20', number_of_colors_needed)
			# Create a dictionary mapping integers to hex color codes
			for color_i, cell_id in enumerate(unique_cell_ids):
				color = cmap(color_i)[:3]  # Extract RGB values
				color_dict[cell_id] = color
			for i in range(start, end, 1):
				phase = phase_stack[i, :, :]
				cells_time = cells_df[cells_df['time_index'] == i]
				if len(cells_time) >= 1:
					for index, cell in cells_time.iterrows():
						(y_loc, x_loc) = cell['centroids']
						color = color_dict[cell['cell_id']]
						axs[i - start].imshow(phase, cmap='gray')
						axs[i - start].scatter(x_loc, y_loc, color=color, s=5)
						axs[i - start].set_yticks([])
						axs[i - start].set_xticks([])
						axs[i - start].set_xlabel(f"{i}", fontsize=8)

	patches = []
	if len(color_dict.items()) >= 1:
		for label, color in color_dict.items():
			patches.append(mpatches.Patch(color=color, label=f"Label {label}"))
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

	plt.show(fig)
	plt.close(fig)


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


