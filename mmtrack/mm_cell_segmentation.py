import os
import re
import numpy as np
import tifffile
import warnings
from skimage import segmentation, morphology, measure, filters
from scipy import ndimage as ndi
import networkx as nx

def segment_chnl_stack(path_to_phase_channel_stack,
					   output_path,
					   OTSU_threshold=1.5,
					   first_opening=5,
					   distance_threshold=3,
					   second_opening_size=1,
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
											min_cell_area,
											max_cell_area,
											small_merge_area_threshold)

		unstacked_seg_image = unstacked_seg_image.astype("uint8")
		unstacked_seg_filename = f'mask_t_{time:03d}_{filename}'
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
                  min_cell_area=200,
                  max_cell_area=700,
                  small_merge_area_threshold=50,
                  min_axis_ratio=0.04):
    """Segments an image with size and shape filtering, and improved structure."""

    # 1. Thresholding and Initial Morphological Operations
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresh = filters.threshold_otsu(image)
        thresholded = image > OTSU_threshold * thresh
        morph = morphology.binary_opening(thresholded, morphology.disk(first_opening))

        if np.amax(morph) == 0:
            return np.zeros_like(image)

        distance = ndi.distance_transform_edt(morph)
        distance_thresh = distance >= distance_threshold
        distance_opened = morphology.binary_opening(distance_thresh, morphology.disk(second_opening_size))

        labeled, num_labels = morphology.label(distance_opened, connectivity=1, return_num=True)
        if num_labels == 0:
            return np.zeros_like(image)

        markers = morphology.label(labeled, connectivity=1)
        markers = markers.astype(np.int32)
        if np.amax(markers) == 0:
            return np.zeros_like(image)

    except Exception as e:
        print(f"Error in initial processing: {e}")
        return np.zeros_like(image)
        
    
    # 2. Watershed Segmentation
    thresholded_watershed = thresholded
    image_float = image.astype(np.float64)
    try:
        markers[thresholded_watershed == 0] = -1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labeled_image = segmentation.random_walker(-1 * image_float, markers)
        labeled_image[labeled_image == -1] = 0
    except Exception as e:
        print(f"Error in watershed segmentation: {e}")
        return np.zeros_like(image)

    # 3. Bounding Box Merging (Small Regions)
    regions = measure.regionprops(labeled_image)
    if regions:
        bboxes = np.array([region.bbox for region in regions])
        areas = np.array([region.area for region in regions])
        graph = nx.Graph()

        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes):
                if i != j and overlap(bbox1, bbox2) and areas[i] < small_merge_area_threshold and areas[j] < small_merge_area_threshold:
                    graph.add_edge(regions[i].label, regions[j].label)

        for component in nx.connected_components(graph):
            if len(component) > 1:
                merged_mask = np.isin(labeled_image, list(component))
                new_label = np.max(labeled_image) + 1
                labeled_image[merged_mask] = new_label
                for label in component:
                    labeled_image[labeled_image == label] = 0

        labeled_image, _ = morphology.label(labeled_image, connectivity=1, return_num=True)

    # 4. Shape Filtering (Axis Ratio)
    filtered_labeled_image = np.zeros_like(labeled_image)
    regions = measure.regionprops(labeled_image)

    for region in regions:
        if region.axis_major_length > 0:
            axis_ratio = region.axis_minor_length / region.axis_major_length
            if (axis_ratio > min_axis_ratio) | (region.axis_minor_length > 3.5):
                filtered_labeled_image[region.coords[:, 0], region.coords[:, 1]] = region.label

    labeled_image = filtered_labeled_image

    # 5. Size Filtering (Area)
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


