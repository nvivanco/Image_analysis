import os
from typing import Tuple
from skimage import segmentation, morphology
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import pandas as pd
import cell_class_from_mm3 as cell



def cells2df(cells, num_time_frames=2,
             columns=['fov', 'peak', 'birth_label', 'parent', 'daughters', 'birth_time', 'times',
                      'labels', 'bboxes', 'areas', 'lengths', 'widths', 'orientations', 'centroids', 'fl_area_avgs', 'is_active']) -> pd.DataFrame:
    """Converts cell data to a DataFrame, handling potential issues."""

    cells_dict = {cell_id: vars(cell) for cell_id, cell in cells.items()}
    df = pd.DataFrame(cells_dict).transpose()
    final_cells_pd = df[columns]
    final_cells_pd = final_cells_pd.sort_values(by=["fov", "peak", "birth_time", "birth_label"])
    final_cells_pd.reset_index(inplace=True)
    final_cells_pd.rename(columns={'index': 'cell_id',
                                   'abs_times': 'abs_times_(sec)',
                                   'times': 'time_index',
                                   'birth_time': 'birth_time_index',
                                   'areas': 'areas_(pxls^2)',
                                   'lengths': 'lengths_(pxls)',
                                   'widths': 'widths_(pxls)'}, inplace=True)

    time_index_lengths = final_cells_pd['time_index'].str.len()
    filtered_cells_pd = final_cells_pd[time_index_lengths > num_time_frames]


    def create_time_points(row):
        n_times = len(row['time_index'])
        data = {
            'cell_id': [row['cell_id']] * n_times,
            'time_index': row['time_index'],
            'is_active': [row['is_active']] * n_times
        }

        # Handle list and single-value columns consistently
        for col in ['daughters', 'parent', 'labels', 'bboxes', 'areas_(pxls^2)', 'lengths_(pxls)', 'widths_(pxls)', 'orientations', 'centroids', 'fl_area_avgs']:
            if isinstance(row[col], list):
                data[col] = row[col]
            else:  # Single value
                data[col] = [row[col]] * n_times # Replicate the single value

        return pd.DataFrame(data)

    time_point_df = filtered_cells_pd.apply(create_time_points, axis=1)

    # Concatenate the DataFrames efficiently
    time_point_df = pd.concat(time_point_df.tolist(), ignore_index=True)

    # Explode only the necessary column (if it's still a list after the previous steps)
    if 'fl_area_avgs' in time_point_df.columns and isinstance(time_point_df['fl_area_avgs'].iloc[0], list):
        time_point_df = time_point_df.explode('fl_area_avgs')

    if time_point_df.empty:
        print("Warning: No cells found after filtering by num_time_frames.")
        return time_point_df

    # Unpack bboxes (handle potential missing bboxes)
    if 'bbox' in time_point_df.columns and isinstance(time_point_df['bbox'].iloc[0], list):
        time_point_df[['bb_xLeft', 'bb_yTop', 'bb_width', 'bb_height']] = pd.DataFrame(time_point_df['bbox'].tolist(), index=time_point_df.index)
        time_point_df.drop(['bbox'], axis=1, inplace=True)
    elif 'bbox' in time_point_df.columns: # If bbox is a single value, unpack accordingly
        time_point_df[['bb_xLeft', 'bb_yTop', 'bb_width', 'bb_height']] = pd.DataFrame([time_point_df['bbox'].tolist()], index=time_point_df.index)
        time_point_df[['bb_xLeft', 'bb_yTop', 'bb_width', 'bb_height']] = time_point_df[['bb_xLeft', 'bb_yTop', 'bb_width', 'bb_height']].apply(pd.Series)
        time_point_df.drop(['bbox'], axis=1, inplace=True)


    return time_point_df


def find_cell_intensities(path_to_original_stack,
                          path_to_segmented_stack,
                          cells, start_frame,
                          midline=False):
    """
    Finds fluorescent information for cells. All the cells in cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    """

    # Load fluorescent images and segmented images for this microfluidic channel

    fl_stack = tifffile.imread(path_to_original_stack)
    seg_stack = tifffile.imread(path_to_segmented_stack)



    # Loop through cells
    for cell in cells.values():
        # give this cell two lists to hold new information
        cell.fl_tots = []  # total fluorescence per time point
        cell.fl_area_avgs = []
        cell.fl_vol_avgs = []  # avg fluorescence per unit volume by timepoint

        if midline:
            cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - start_frame])
            fl_image_masked[seg_stack[t - start_frame] != cell.labels[n]] = 0

            # append total flourescent image
            cell.fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            cell.fl_area_avgs.append(np.sum(fl_image_masked) / cell.areas[n])
            cell.fl_vol_avgs.append(np.sum(fl_image_masked) / cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - start_frame])
                bin_mask[bin_mask != cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                # med_mask[med_dist < np.floor(cap_radius/2)] = 0
                # print(img_fluo[med_mask])
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    cell.mid_fl.append(np.nan)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.

def make_lineage_chnl_stack(path_to_stack: str,
                            labeled_stack: str,
                            fov_id: int,
                            peak_id: int,
                            pxl2um: float,
                            start_frame: int = 0) -> dict:
    """
    Create the lineage for a set of segmented images for one channel. Start by making the regions in the first time points potenial cells.
    Go forward in time and map regions in the timepoint to the potential cells in previous time points, building the life of a cell.
    Used basic checks such as the regions should overlap, and grow by a little and not shrink too much. If regions do not link back in time, discard them.
    If two regions map to one previous region, check if it is a sensible division event.

    Parameters
    ----------
    fov_id: int
    peak_id: int
    pxl2um: float, pixels to microns conversion

    Returns
    -------
    cells : dict
        A dictionary of all the cells from this lineage, divided and undivided

    """

    # load in parameters
    # if leaf regions see no action for longer than this, drop them
    lost_cell_time = 20

    image_data_seg = tifffile.imread(labeled_stack)

    cells = {}
    cell_leaves = []
    previous_cells = {}
    previous_regions = None

    # go through regions by timepoint and build lineages
    # timepoints start with the index of the first image
    for t in range(start_frame, image_data_seg.shape[0]):  # Start from start_time

        cell_leaves, cells = prune_leaves(cell_leaves, cells, lost_cell_time, t)

        current_regions = process_frame(image_data_seg[t], previous_regions)

        if not current_regions:  # Check if current_regions is empty
            continue  # Skip to the next time point

        if not previous_cells:  # Check if previous_cells is empty (only at the *real* start)
            if t == start_frame:  # Initialize only at the specified start time
                for region in current_regions:
                    cell_id = create_cell_id(region, t, fov_id, peak_id)
                    cells[cell_id] = cell.Cell(pxl2um, cell_id, region, t, parent_id=None)
                    cell_leaves.append(cell_id)
                    previous_cells[cell_id] = cells[cell_id]
                continue  # Move to the next frame
            else:
                continue  # Skip if previous_cells is empty but it's not the start time

        matches, unmatched_current, unmatched_previous, cell_leaves = link_regions(
            current_regions, previous_cells, t, cells, cell_leaves,
            fov_id, peak_id, pxl2um, cost_threshold=150, lookback_window=3
        )
        #visualize_matches(image_data_seg, t, matches, unmatched_current, unmatched_previous, previous_cells, current_regions)

        for cell_id in matches:
            cells[cell_id].last_meaningful_update = t
        previous_cells = cells

        # Handle Growth and Division
        for prev_cell_id in unmatched_previous:
            mother_cell = cells[prev_cell_id]
            daughter_regions = []

            for current_region in current_regions:  # Iterate through the current_regions
                centroid_dist = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(current_region.centroid))
                if centroid_dist < 100:  # Adjust distance threshold
                    daughter_regions.append(current_region)

            if len(daughter_regions) == 2:  # Found two potential daughters

                growth1 = check_growth_by_region(mother_cell, daughter_regions[0])
                growth2 = check_growth_by_region(mother_cell, daughter_regions[1])

                if growth1 or growth2:  # At least one daughter is simple growth
                    if growth1:
                        cells[prev_cell_id].grow(daughter_regions[0], t)
                        cells[prev_cell_id].last_meaningful_update = t
                        try:
                            unmatched_current.remove(current_regions.index(daughter_regions[0]))
                        except ValueError:
                            pass
                    elif growth2:
                        cells[prev_cell_id].grow(daughter_regions[1], t)
                        cells[prev_cell_id].last_meaningful_update = t
                        try:
                            unmatched_current.remove(current_regions.index(daughter_regions[1]))
                        except ValueError:
                            pass
                    continue  # Move to the next mother cell

                else:  # No simple growth; check for division
                    print(f"Processing timepoint t = {t}")
                    if check_division(mother_cell, daughter_regions[0], daughter_regions[1]):
                        print('check division')
                        daughter1_id, daughter2_id, cells = divide_cell(
                            daughter_regions[0], daughter_regions[1], t, peak_id, fov_id, pxl2um, cells,
                            prev_cell_id
                        )
                        cells[prev_cell_id].last_meaningful_update = t
                        cells[daughter1_id].last_meaningful_update = t
                        cells[daughter2_id].last_meaningful_update = t
                        cell_leaves.remove(prev_cell_id)
                        cell_leaves.extend([daughter1_id, daughter2_id])
                        try:
                            unmatched_current = set(unmatched_current) - set(
                                [current_regions.index(region) for region in daughter_regions])
                        except ValueError:
                            pass

        # Handle remaining unmatched current regions as new cells (if not part of a division)
        for region in current_regions:  # Iterate through the current_regions
            if region.label not in {cell.labels[-1] for cell in cells.values() if
                                    cell.times[-1] == t}:  # Check if label exists
                cell_id = create_cell_id(region, t, fov_id, peak_id)
                cells[cell_id] = cell.Cell(pxl2um, cell_id, region, t, parent_id=None)
                cells[cell_id].last_meaningful_update = t  # New cell
                cell_leaves.append(cell_id)
                previous_cells[cell_id] = cells[cell_id]

        previous_regions = current_regions

        # Handle New Cells in the Last Frame (Adjusted)
    last_frame_index = image_data_seg.shape[0] - 1  # Correct way to get last index
    last_frame_regions = process_frame(image_data_seg[last_frame_index])  # Use process frame function
    tracked_last_frame_regions = {cell.labels[-1] for cell in cells.values() if cell.times[-1] == last_frame_index}

    for region in last_frame_regions:
        if region.label not in tracked_last_frame_regions:
            cell_id = create_cell_id(region, last_frame_index, fov_id, peak_id)
            cells[cell_id] = cell.Cell(pxl2um, cell_id, region, last_frame_index, parent_id=None)
            cell_leaves.append(cell_id)

    # return the dictionary with all the cells
    return cells


def visualize_matches(image_data_seg, t, matches, unmatched_current, unmatched_previous, previous_cells, current_regions):
    """Visualizes matched and unmatched regions between frames."""

    if t > 0:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot Previous Frame (t-1)
        axes[0].imshow(image_data_seg[t - 1], cmap='gray')
        axes[0].set_title(f"Previous Frame (t={t - 1})")

        for prev_cell_id, current_region_index in matches.items():
            prev_cell = previous_cells[prev_cell_id]
            for region in prev_cell.regions:
                bbox = region.bbox
                rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                         linewidth=1, edgecolor='r', facecolor='none')
                axes[0].add_patch(rect)
            previous_centroid = np.array(prev_cell.centroids[-1])
            axes[0].plot(previous_centroid[1], previous_centroid[0], 'ro', label="Matched Previous")

        for prev_cell_id in unmatched_previous:
            prev_cell = previous_cells[prev_cell_id]
            for region in prev_cell.regions:
                bbox = region.bbox
                rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                         linewidth=1, edgecolor='b', facecolor='none')
                axes[0].add_patch(rect)
            previous_centroid = np.array(prev_cell.centroids[-1])
            axes[0].plot(previous_centroid[1], previous_centroid[0], 'bo', label="Unmatched Previous")
        axes[0].legend()
        axes[0].set_aspect('equal', adjustable='box')

        # Plot Current Frame (t)
        axes[1].imshow(image_data_seg[t], cmap='gray')
        axes[1].set_title(f"Current Frame (t={t})")

        for prev_cell_id, current_region_index in matches.items():
            current_region = current_regions[current_region_index]
            bbox = current_region.bbox
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                     linewidth=1, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            current_centroid = np.array(current_region.centroid)
            axes[1].plot(current_centroid[1], current_centroid[0], 'ro', label="Matched Current")

        for i in unmatched_current:
            current_region = current_regions[i]
            bbox = current_region.bbox
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                     linewidth=1, edgecolor='g', facecolor='none')
            axes[1].add_patch(rect)
            current_centroid = np.array(current_region.centroid)
            axes[1].plot(current_centroid[1], current_centroid[0], 'go', label="Unmatched Current")

        axes[1].legend()
        axes[1].set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

    elif t == 0:
        plt.figure(figsize=(10, 8))
        plt.imshow(image_data_seg[t], cmap='gray')
        plt.title(f"Current Frame (t={t})")

        for i in unmatched_current:
            current_region = current_regions[i]
            bbox = current_region.bbox
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                     linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
            current_centroid = np.array(current_region.centroid)
            plt.plot(current_centroid[1], current_centroid[0], 'go', label="Unmatched Current")

        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()


def plot_regions(
    seg_data: np.ndarray,
    cells: dict,
    t: int,
    ax: plt.Axes,
    cmap: str,
    vmin: float = 0.5,
    vmax: float = 100) -> plt.Axes:
    """Plots segmented cells from the cells dictionary."""


    for cell in cells.values():
        if not cell.is_active:  # Skip inactive cells
            continue

        if t in cell.times:  # Check if the cell exists at this time point
            n = cell.times.index(t) # Get the index for the time point
            region = cell.regions[n] # Access the correct region
            rescaled_color_index = region.centroid[0] / seg_data.shape[0] * vmax
            seg_relabeled[seg_relabeled == region.label] = (
                int(rescaled_color_index) - 0.1
            )

    ax.imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    return ax


def prune_leaves(cell_leaves, cells, lost_cell_time, t):
    if not cell_leaves:
        return cell_leaves, cells

    leaves_to_deactivate = []
    for leaf_id in cell_leaves:
        if 'last_meaningful_update' not in cells[leaf_id].__dict__:  # Check if the attribute exists
            cells[leaf_id].last_meaningful_update = cells[leaf_id].times[-1]

        if t - cells[leaf_id].last_meaningful_update > lost_cell_time:
            leaves_to_deactivate.append(leaf_id)

    for leaf_id in leaves_to_deactivate:
        cell_leaves.remove(leaf_id)
        if leaf_id in cells:
            cells[leaf_id].is_active = False

    return cell_leaves, cells


def divide_cell(
    region1,
    region2,
    t: int,
    peak_id: int,
    fov_id: int,
    pxl2um: float,
    cells: dict[str, cell.Cell],
    leaf_id: str) -> Tuple[str, str, dict]:
    """
    Create two new cells and divide the mother

    Parameters
    ----------
    region1: RegionProperties object
        first region
    region2: RegionProperties object
        second region
    t: int
        current time step
    peak_id: int
        current peak (trap) id
    fov_id: int
        current FOV id
    pxl2um: float
        pixel to micron conversion factor
    cells: dict[str, Cell]
        dictionary of cell objects
    leaf_id: str
        cell id of current leaf

    Returns
    -------
    daughter1_id: str
        cell id of 1st daughter
    daughter2_id: str
        cell id of 2nd daughter
    cells: dict[str, Cell]
        updated dictionary of Cell objects
    """

    daughter1_id = create_cell_id(region1, t, peak_id, fov_id)
    daughter2_id = create_cell_id(region2, t, peak_id, fov_id)
    cells[daughter1_id] = cell.Cell(
        pxl2um,
        daughter1_id,
        region1,
        t,
        parent_id=leaf_id,
    )
    cells[daughter2_id] = cell.Cell(
        pxl2um,
        daughter2_id,
        region2,
        t,
        parent_id=leaf_id,
    )
    print('daughters and parent names assigned')
    cells[leaf_id].divide(cells[daughter1_id], cells[daughter2_id], t)

    return daughter1_id, daughter2_id, cells


def process_frame(seg_image, previous_regions=None):
    """Segments and relabels regions for a single frame."""
    current_regions = regionprops(seg_image)
    full_image_shape = seg_image.shape

    if previous_regions:
        current_regions = relabel_regions(current_regions, previous_regions, full_image_shape)

    return current_regions


def relabel_regions(current_regions, previous_regions, full_image_shape):  # Add full_image_shape
    """Relabels regions using IoU and the Hungarian algorithm (Corrected)."""

    if not previous_regions:
        for i, region in enumerate(current_regions):
            region.label = i + 1
        return current_regions

    cost_matrix = np.zeros((len(current_regions), len(previous_regions)))

    for i, current_region in enumerate(current_regions):
        for j, previous_region in enumerate(previous_regions):
            # Create masks of the full image shape
            current_mask = np.zeros(full_image_shape, dtype=bool)
            previous_mask = np.zeros(full_image_shape, dtype=bool)

            # Set mask values based on region pixels
            y_c, x_c = current_region.centroid
            y_c, x_c = int(y_c), int(x_c)
            y_p, x_p = previous_region.centroid
            y_p, x_p = int(y_p), int(x_p)

            # Get bounding box coordinates for the current and previous regions
            min_y_c, min_x_c, max_y_c, max_x_c = current_region.bbox
            min_y_p, min_x_p, max_y_p, max_x_p = previous_region.bbox

            # Ensure coordinates are within image bounds
            min_y_c, min_x_c = max(0, min_y_c), max(0, min_x_c)
            max_y_c, max_x_c = min(full_image_shape[0], max_y_c), min(full_image_shape[1], max_x_c)

            min_y_p, min_x_p = max(0, min_y_p), max(0, min_x_p)
            max_y_p, max_x_p = min(full_image_shape[0], max_y_p), min(full_image_shape[1], max_x_p)


            # Set mask values within bounding boxes
            current_mask[min_y_c:max_y_c, min_x_c:max_x_c][current_region.image.astype(bool)] = True
            previous_mask[min_y_p:max_y_p, min_x_p:max_x_p][previous_region.image.astype(bool)] = True

            intersection = np.sum(current_mask & previous_mask)
            union = np.sum(current_mask | previous_mask)

            iou = intersection / union if union > 0 else 0
            cost_matrix[i, j] = -iou  # Maximize IoU

    # ... (Rest of the Hungarian algorithm and label assignment remain the same)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    relabelled_regions = []
    assigned_previous_labels = set()
    for i, j in zip(row_ind, col_ind):
        current_region = current_regions[i]
        previous_region = previous_regions[j]
        current_region.label = previous_region.label
        assigned_previous_labels.add(previous_region.label)
        relabelled_regions.append(current_region)

    # Handle unmatched current regions
    unmatched_current_regions = [
        current_regions[i] for i in range(len(current_regions)) if i not in row_ind
    ]
    next_available_label = max(
        [region.label for region in previous_regions], default=0
    ) + 1
    for region in unmatched_current_regions:
        region.label = next_available_label
        next_available_label += 1
        relabelled_regions.append(region)

    return relabelled_regions


def link_regions(current_regions, previous_cells, t, cells, cell_leaves, fov_id, peak_id, pxl2um, cost_threshold=150, lookback_window=3):
    """Links regions between frames, including lookback matching."""

    n_current = len(current_regions)
    n_previous = len(previous_cells)

    if n_previous == 0:  # First frame
        for current_region in current_regions:
            cell_id = create_cell_id(current_region, t, fov_id, peak_id)
            cells[cell_id] = cell.Cell(pxl2um, cell_id, current_region, t, parent_id=None)
            cell_leaves.append(cell_id)
            previous_cells[cell_id] = cells[cell_id]
        return {}, set(), set(), cell_leaves

    cost_matrix = np.zeros((n_current, n_previous), dtype=float)

    for i, current_region in enumerate(current_regions):
        for j, (prev_cell_id, prev_cell_object) in enumerate(previous_cells.items()):
            prev_region = prev_cell_object.regions[-1]
            centroid_dist = np.linalg.norm(np.array(current_region.centroid) - np.array(prev_region.centroid))
            area_diff = abs(current_region.area - prev_region.area)
            cost = centroid_dist + 0.1 * area_diff  # Adjust weight (0.1) as needed
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            prev_cell_id = list(previous_cells.keys())[j]
            matches[prev_cell_id] = i
            cells[prev_cell_id].grow(current_regions[i], t)

    unmatched_current = set(range(n_current)) - set(row_ind)
    unmatched_previous = set(previous_cells) - set([list(previous_cells.keys())[k] for k in col_ind])

    # Lookback Matching
    for i in list(unmatched_current):  # Iterate over a copy
        current_region = current_regions[i]
        best_match_cell_id = None
        min_cost = float('inf')

        for frame_offset in range(1, lookback_window + 1):
            lookback_time = t - frame_offset
            if lookback_time < 0:  # Handle edge case: no previous frame
                break

            potential_previous_cells = {
                cell_id: cell for cell_id, cell in cells.items()
                if cell.times[-1] == lookback_time and cell.fov == fov_id and cell.peak == peak_id
            }

            if not potential_previous_cells:
                continue

            for prev_cell_id, prev_cell_object in potential_previous_cells.items():
                prev_region = prev_cell_object.regions[-1]
                centroid_dist = np.linalg.norm(
                    np.array(current_region.centroid) - np.array(prev_region.centroid)
                )
                area_diff = abs(current_region.area - prev_region.area)
                cost = centroid_dist + 0.1 * area_diff  # Your cost function
                if cost < min_cost:
                    min_cost = cost
                    best_match_cell_id = prev_cell_id

        if best_match_cell_id is not None and min_cost < cost_threshold * 2:  # Increased threshold
            cells[best_match_cell_id].grow(current_region, t)
            unmatched_current.remove(i)
            if best_match_cell_id in unmatched_previous:
                unmatched_previous.remove(best_match_cell_id)

    # Handle remaining unmatched current regions as new cells
    for i in unmatched_current:
        current_region = current_regions[i]
        cell_id = create_cell_id(current_region, t, fov_id, peak_id)
        cells[cell_id] = cell.Cell(pxl2um, cell_id, current_region, t, parent_id=None)
        cell_leaves.append(cell_id)
        previous_cells[cell_id] = cells[cell_id]

    return matches, unmatched_current, unmatched_previous, cell_leaves
def check_division(mother_cell: cell.Cell,
                   region1, region2,
                   min_area_ratio=0.9,
                   dist_threshold=150) -> int:
    """Checks to see if it makes sense to divide a
    cell into two new cells based on two regions."""

    # Basic Area Check
    total_daughter_area = region1.area + region2.area

    if not (mother_cell.areas[-1] * min_area_ratio <= total_daughter_area):
        print('area check failed')
        return False  # Area check failed

    # Proximity Check
    centroid_dist1 = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(region1.centroid))
    centroid_dist2 = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(region2.centroid))
    daughter_dist = np.linalg.norm(np.array(region1.centroid) - np.array(region2.centroid))

    print('mother and region 1 distance')
    print(centroid_dist1)
    print('mother and region 2 distance')
    print(centroid_dist2)
    print('region 1 and region 2 distance')
    print(daughter_dist)

    if centroid_dist1 > dist_threshold or centroid_dist2 > dist_threshold or daughter_dist > dist_threshold:
        print('proximity check failed')
        return False  # Proximity check failed

    return True  # All checks passed; it's a division


def create_cell_id(
    region, t: int, peak: int, fov: int, experiment_name: str = '') -> str:
    """Make a unique cell id string for a new cell
    Parameters
    ----------
    region: regionprops object
        region to initialize cell from
    t: int
        time
    peak: int
        peak id
    fov: int
        fov id
    experiment_name: str
        experiment label

    Returns
    -------
    cell_id: str
        string for cell ID
    """
    cell_id = f"{experiment_name}f{fov:02d}p{peak:04d}t{t:04d}r{region.label:02d}"

    return cell_id


def check_growth_by_region(cell: cell.Cell, region) -> bool:
    """Checks to see if it makes sense
    to grow a cell by a particular region

    Parameters
    ----------
    cell: Cell object
        Cell object currently tracked
    region: RegionProps object
        regionprops object containing area attributes

    Returns
    -------
    bool
        True if it makes sense to grow the cell
    """
    # load parameters in ratios
    max_growth_length = 1.3
    min_growth_length = 0.8
    max_growth_area = 1.3
    min_growth_area = 0.8

    # check if length is not too much longer
    if cell.lengths[-1] * max_growth_length < region.major_axis_length:
        return False

    # check if it is not too short (cell should not shrink really)
    if cell.lengths[-1] * min_growth_length > region.major_axis_length:
        return False

    # check if area is not too great
    if cell.areas[-1] * max_growth_area < region.area:
        return False

    # check if area is not too small
    if cell.lengths[-1] * min_growth_area > region.area:
        return False

    # check if y position of region is within the bounding box of previous region
    lower_bound = cell.bboxes[-1][0]
    upper_bound = cell.bboxes[-1][2]
    if lower_bound > region.centroid[0] or upper_bound < region.centroid[0]:
        return False

    # return true if you get this far
    return True