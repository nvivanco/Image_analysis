from typing import Tuple
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import cell_class_from_mm3 as cell_class
import plot_cells



def cells2df(cells, num_time_frames=2,
             columns=['fov', 'peak', 'birth_label', 'parent', 'daughters', 'birth_time', 'times',
                      'labels', 'bboxes', 'areas', 'lengths', 'widths', 'orientations', 'centroids',
                      'fl_area_avgs', 'mid_intensity','fl_vol_avgs', 'fl_tots', 'masks', 'is_active']) -> pd.DataFrame:
    """Converts cell data to a DataFrame, handling potential issues."""

    cells_dict = {cell_id: vars(cell) for cell_id, cell in cells.items()}
    df = pd.DataFrame(cells_dict).transpose()
    final_cells_pd = df[columns]
    final_cells_pd = final_cells_pd.sort_values(by=['fov', 'peak', 'birth_time', 'birth_label'])
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

    if filtered_cells_pd.empty:
        print(f"Warning: No cells found after filtering (num_time_frames = {num_time_frames}).")
        return pd.DataFrame(columns=final_cells_pd.columns)

    def create_time_points(row):
        n_times = len(row['time_index'])
        data = {
            'cell_id': [row['cell_id']] * n_times,
            'time_index': row['time_index'],
            'is_active': [row['is_active']] * n_times
        }

        # Handle list and single-value columns consistently
        for col in ['daughters', 'parent', 'labels', 'bboxes', 'areas_(pxls^2)', 'lengths_(pxls)',
                    'widths_(pxls)', 'orientations', 'centroids', 'fl_area_avgs', 'mid_intensity',
                    'fl_vol_avgs', 'fl_tots', 'masks']:
            if isinstance(row[col], list):
                data[col] = row[col]
            else:  # Single value
                data[col] = [row[col]] * n_times # Replicate the single value

        return pd.DataFrame(data)

    time_point_df = filtered_cells_pd.apply(create_time_points, axis=1)
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

def _midline_intensity(region, image):
    """
    Calculates the intensity along the middle horizontal axis of a cell.

    Args:
        region: A skimage.measure._regionprops.RegionProperties object.
        image: The masked image (NumPy array) containing the fluorescence data.

    Returns:
        float: The average intensity along the horizontal midline, or NaN if the region is invalid.
    """
    if region is None:
        return np.nan

    centroid = region.centroid
    center_row = int(centroid[0])  # Get the integer row coordinate

    if 0 <= center_row < image.shape[0]:
        horizontal_line = image[center_row, :]
        if horizontal_line.size > 0:
            return np.nanmean(horizontal_line)
        else:
            return np.nan
    else:
        return np.nan #centroid out of image bounds

def find_cell_intensities(path_to_stack, cells):
    """
    Finds fluorescent information for cells. All the cells in cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    The cell objects in the original dictionary will be updated.
    """

    # Load fluorescent images and segmented images for this microfluidic channel

    fl_stack = tifffile.imread(path_to_stack)

    # Loop through cells
    for cell_id, cell in cells.items():
        cell.masks = [] # save cell masks in cell_dict
        cell.fl_tots = []  # total pixel intensity per time point
        cell.fl_area_avgs = []
        cell.fl_vol_avgs = []  # avg pixel intensity per unit volume by timepoint
        cell.mid_intensity = []  # avg pixel intensity of horizontal midline

        # and the time points that make up this cell's life
        for t in cell.times:
            n = cell.times.index(t)
            image = np.copy(fl_stack[t])
            region = cell.regions[n]
            cell_mask = plot_cells.mask_from_region(region, image)
            masked_image = image * cell_mask
            middle_intensity = _midline_intensity(region, image)

            cell.masks.append(cell_mask)
            cell.fl_tots.append(np.sum(masked_image))
            cell.mid_intensity.append(middle_intensity)
            cell.fl_area_avgs.append(np.sum(masked_image) / cell.areas[n])
            cell.fl_vol_avgs.append(np.sum(masked_image) / cell.volumes[n])

def make_lineage_chnl_stack(labeled_stack: str,
                            fov_id: int,
                            peak_id: int,
                            pxl2um: float,
                            start_frame: int = 0,
                            lookback_window = 10,
                            lookforward_window = 10) -> dict:
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

        if not current_regions:
            continue

        # Lost tracked regions are still present here
        #plot_region_of_interest(image_data_seg, t, current_regions)

        if not previous_cells:
            if t == start_frame:  # Initialize only at the specified start time
                for region in current_regions:
                    cell_id = create_cell_id(region, t, fov_id, peak_id)
                    cells[cell_id] = cell_class.Cell(pxl2um, cell_id, region, t, parent_id=None)
                    cell_leaves.append(cell_id)
                    previous_cells[cell_id] = cells[cell_id]
                continue  # Move to the next frame
            else:
                continue  # Skip if previous_cells is empty but it's not the start time

        # Only track active cells
        active_previous_cells = {
            cell_id: cell for cell_id, cell in previous_cells.items() if cell.is_active
        }


        if active_previous_cells:
            matches, unmatched_current, unmatched_previous, cell_leaves = link_regions(
                current_regions, active_previous_cells, t, cells, cell_leaves,  # Pass active cells ONLY
                fov_id, peak_id, pxl2um, cost_threshold=150
            )

            for cell_id in matches:
                cells[cell_id].last_meaningful_update = t
            previous_cells = cells

            # extract regions of matched cells
            #if t < 37:
                #for cell_id, cell in previous_cells.items():
                #    plot_region_of_interest(image_data_seg, t, cell.regions)


            # Handle Unmatched with lookback and lookforward
            active_unmatched_previous = {
                cell_id: active_previous_cells[cell_id] for cell_id in unmatched_previous if
                cell_id in active_previous_cells
            }

            #check cell_leaves for lost cells in unmatched_previous
            for cell_id in unmatched_previous:
                cell = previous_cells[cell_id]
                if (t < 37):
                    plot_region_of_interest(image_data_seg, t, [cell.regions[-1]])

            #check cell_leaves for lost cells in cell leaves
            # for cell_id in cell_leaves:
            #     last_cell = cells[cell_id]
            #     if (t < 37):
            #         plot_region_of_interest(image_data_seg, (t-1), [last_cell.regions[-1]])

            # extract regions of unmatched current cells
            # unmatched_current_regions = [current_regions[i] for i in unmatched_current]
            # if (t < 37) & (len(unmatched_current_regions)> 0):
            #     plot_region_of_interest(image_data_seg, t, unmatched_current_regions)


            for i in list(unmatched_current):
                current_region = current_regions[i]
                # Check if the current region has already been assigned to an active cell
                already_assigned = False
                for cell in cells.values():
                    if cell.is_active and t > 0 and current_region.label in [r.label for r in cell.regions if
                                                                             cell.times[-1] == (t - 1)]:
                        already_assigned = True

                        break
                if already_assigned:
                    continue  # If it has been assigned, skip it

                # Lookback Matching
                best_lookback_match = None
                min_lookback_cost = float('inf')

                for frame_offset in range(1, lookback_window + 1):
                    lookback_time = t - frame_offset
                    if lookback_time < 0:
                        break

                    potential_lookback_cells = {
                        cell_id: cell
                        for cell_id, cell in cells.items()
                        if cell.times[
                               -1] == lookback_time and cell.fov == fov_id and cell.peak == peak_id and cell.is_active
                        # Check for active cells
                    }

                    for prev_cell_id, prev_cell in potential_lookback_cells.items():
                        prev_region = prev_cell.regions[-1]
                        centroid_dist = np.linalg.norm(
                            np.array(current_region.centroid) - np.array(prev_region.centroid)
                        )

                        area_diff = abs(current_region.area - prev_region.area)
                        cost = centroid_dist + 0.1 * area_diff
                        if cost < min_lookback_cost:
                            min_lookback_cost = cost
                            best_lookback_match = prev_cell_id

                if best_lookback_match and min_lookback_cost < 50 * 2:  # Increased threshold
                    cells[best_lookback_match].grow(current_region, t)
                    unmatched_current.remove(i)
                    continue  # Skip to the next unmatched current region

                # Lookforward Matching
                best_lookforward_match = None
                min_lookforward_cost = float('inf')

                for frame_offset in range(1, lookforward_window + 1):
                    lookforward_time = t + frame_offset
                    if lookforward_time >= image_data_seg.shape[0]:
                        break

                    potential_lookforward_cells = {
                        cell_id: cell
                        for cell_id, cell in cells.items()
                        if cell.times[
                               -1] == lookforward_time and cell.fov == fov_id and cell.peak == peak_id and cell.is_active
                        # Check for active cells
                    }

                    for next_cell_id, next_cell in potential_lookforward_cells.items():
                        next_region = next_cell.regions[-1]
                        centroid_dist = np.linalg.norm(
                            np.array(current_region.centroid) - np.array(next_region.centroid)
                        )
                        area_diff = abs(current_region.area - next_region.area)
                        cost = centroid_dist + 0.1 * area_diff
                        if cost < min_lookforward_cost:
                            min_lookforward_cost = cost
                            best_lookforward_match = next_cell_id

                if best_lookforward_match and min_lookforward_cost < 50 * 2:
                    current_cell_id = None
                    for cell_id, cell in previous_cells.items():
                        if cell.labels[-1] == current_regions[i].label and cell.times[-1] == (
                                t - 1) and cell.is_active:  # Check for active cells
                            current_cell_id = cell_id
                            break
                    if current_cell_id:
                        cells[current_cell_id].grow(current_regions[i], t)
                        unmatched_current.remove(i)
                        continue

                # New Cell Creation if No Match Back or Forward in Time
                cell_id = create_cell_id(current_region, t, fov_id, peak_id)
                cells[cell_id] = cell_class.Cell(pxl2um, cell_id, current_region, t, parent_id=None)
                cells[cell_id].last_meaningful_update = t
                cell_leaves.append(cell_id)
                previous_cells[cell_id] = cells[cell_id]

            # Handle Potential Division
            for prev_cell_id in list(active_unmatched_previous):
                mother_cell = cells.get(prev_cell_id)
                if mother_cell is None or not mother_cell.is_active:
                    continue
                daughter_regions = []

                for current_region in current_regions:
                    centroid_dist = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(current_region.centroid))
                    if centroid_dist < 150:  # Adjust distance threshold
                        daughter_regions.append(current_region)

                if len(daughter_regions) == 2:  # Found two potential daughters

                    growth1 = check_growth_by_region(mother_cell, daughter_regions[0])
                    growth2 = check_growth_by_region(mother_cell, daughter_regions[1])

                    if growth1 or growth2:  # At least one daughter is simple growth
                        if growth1:
                            cells[prev_cell_id].grow(daughter_regions[0], t)
                            cells[prev_cell_id].last_meaningful_update = t
                            # Remove the region from unmatched_current
                            label_to_remove = daughter_regions[0].label
                            indices_to_remove = [i for i, region in enumerate(current_regions) if
                                                 region.label == label_to_remove]
                            if indices_to_remove:
                                unmatched_current.discard(indices_to_remove[0])
                        elif growth2:
                            cells[prev_cell_id].grow(daughter_regions[1], t)
                            cells[prev_cell_id].last_meaningful_update = t
                            # Remove the region from unmatched_current
                            label_to_remove = daughter_regions[1].label
                            indices_to_remove = [i for i, region in enumerate(current_regions) if
                                                 region.label == label_to_remove]
                            if indices_to_remove:
                                unmatched_current.discard(indices_to_remove[0])
                        continue  # Move to the next mother cell

                    else:  # No simple growth, check for division
                        if check_division(mother_cell, daughter_regions[0], daughter_regions[1]):
                            daughter1_id, daughter2_id, cells = divide_cell(
                                daughter_regions[0], daughter_regions[1], t, peak_id, fov_id, pxl2um, cells,
                                prev_cell_id
                            )
                            if prev_cell_id not in cell_leaves:
                                continue
                            cells[prev_cell_id].last_meaningful_update = t
                            cells[daughter1_id].last_meaningful_update = t
                            cells[daughter2_id].last_meaningful_update = t
                            cell_leaves.remove(prev_cell_id)
                            cell_leaves.extend([daughter1_id, daughter2_id])
                            # Remove daughter regions from unmatched_current by LABEL
                            labels_to_remove = [region.label for region in daughter_regions]
                            indices_to_remove = [i for i, region in enumerate(current_regions) if
                                                 region.label in labels_to_remove]
                            unmatched_current = unmatched_current - set(indices_to_remove)

        previous_regions = current_regions

    # Handle New Cells in the Last Frame
    last_frame_index = image_data_seg.shape[0] - 1
    last_frame_regions = process_frame(image_data_seg[last_frame_index])
    tracked_last_frame_regions = {cell.labels[-1] for cell in cells.values() if cell.times[-1] == last_frame_index}

    for region in last_frame_regions:
        if region.label not in tracked_last_frame_regions:
            cell_id = create_cell_id(region, last_frame_index, fov_id, peak_id)
            cells[cell_id] = cell_class.Cell(pxl2um, cell_id, region, last_frame_index, parent_id=None)
            cell_leaves.append(cell_id)

    # return the dictionary with all the cells
    return cells

def plot_region_of_interest(image_data_seg, t, regions):
    fig, axes = plt.subplots(1, 1, figsize=(5, 8))
    # Plot region of interest on current frame
    axes.imshow(image_data_seg[t], cmap='gray')
    axes.set_title(f"Current Frame (t={t})")
    for region in regions:
        bbox = region.bbox
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                 linewidth=1, edgecolor='r', facecolor='none')
        axes.add_patch(rect)
        axes.plot(region.centroid[1], region.centroid[0], 'ro', label="region of interest")
    axes.legend()
    axes.set_aspect('equal', adjustable='box')

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
    cells: dict[str, cell_class.Cell],
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
    cells[daughter1_id] = cell_class.Cell(
        pxl2um,
        daughter1_id,
        region1,
        t,
        parent_id=leaf_id,
    )
    cells[daughter2_id] = cell_class.Cell(
        pxl2um,
        daughter2_id,
        region2,
        t,
        parent_id=leaf_id,
    )
    cells[leaf_id].divide(cells[daughter1_id], cells[daughter2_id], t)

    return daughter1_id, daughter2_id, cells


def process_frame(seg_image, previous_regions=None):
    """Segments and relabels regions for a single frame."""
    current_regions = regionprops(seg_image)
    full_image_shape = seg_image.shape

    if previous_regions:
        current_regions = relabel_regions(current_regions, previous_regions, full_image_shape)

    return current_regions


def relabel_regions(current_regions, previous_regions, full_image_shape):
    """Relabels regions using IoU and the Hungarian algorithm."""

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


def link_regions(current_regions, previous_cells, t, cells, cell_leaves, fov_id, peak_id, pxl2um,
                 cost_threshold=150):
    """Links regions between frames, including lookback and look-forward matching and division handling."""

    n_current = len(current_regions)
    n_previous = len(previous_cells)

    if n_previous == 0:  # First frame
        for current_region in current_regions:
            cell_id = create_cell_id(current_region, t, fov_id, peak_id)
            cells[cell_id] = cell_class.Cell(pxl2um, cell_id, current_region, t, parent_id=None)
            cell_leaves.append(cell_id)
            previous_cells[cell_id] = cells[cell_id]
        return {}, set(), set(), cell_leaves

    cost_matrix = np.zeros((n_current, n_previous), dtype=float)

    for i, current_region in enumerate(current_regions):
        for j, (prev_cell_id, prev_cell_object) in enumerate(previous_cells.items()):
            prev_region = prev_cell_object.regions[-1]
            # only track y axis of centroid, centroid is (y, x) tuple
            # y should be getting smaller, since cells keep getting pushed upwards by nascent cells
            change_in_y = prev_region.centroid[0] - current_region.centroid[0]
            centroid_dist = np.linalg.norm(np.array(current_region.centroid) - np.array(prev_region.centroid))
            area_diff = abs(current_region.area - prev_region.area)
            cost = centroid_dist + 0.1 * area_diff  # Adjust weight (0.1) as needed
            if change_in_y < -3:
                penalty_factor = 1 + abs(change_in_y + 3) * 0.2  # Example
                cost += centroid_dist * penalty_factor
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            prev_cell_id = list(previous_cells.keys())[j]

            prev_cell = previous_cells[prev_cell_id]  # Get the previous cell object
            current_region = current_regions[i]

            # Shrinkage and Growth Check
            previous_area = prev_cell.regions[-1].area
            current_area = current_region.area
            change_factor = current_area / previous_area if previous_area > 0 else 0

            if change_factor < 0.7 or change_factor > 1.3:
                continue  # Skip if excessive change

            matches[prev_cell_id] = i
            cells[prev_cell_id].grow(current_region, t)

    unmatched_current = set(range(n_current)) - set(row_ind)
    unmatched_previous = set(previous_cells) - set([list(previous_cells.keys())[k] for k in col_ind])

    return matches, unmatched_current, unmatched_previous, cell_leaves


def check_division(mother_cell: cell_class.Cell,
                   region1, region2,
                   min_area_ratio=0.7,
                   dist_threshold=150) -> int:
    """Checks to see if it makes sense to divide a
    cell into two new cells based on two regions."""

    # Basic Area Check
    total_daughter_area = region1.area + region2.area

    if not (
            mother_cell.areas[-1] * min_area_ratio <= total_daughter_area and
            total_daughter_area <= mother_cell.areas[-1] * 1.3
    ):
        return False

    # Proximity Check
    centroid_dist1 = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(region1.centroid))
    centroid_dist2 = np.linalg.norm(np.array(mother_cell.centroids[-1]) - np.array(region2.centroid))
    daughter_dist = np.linalg.norm(np.array(region1.centroid) - np.array(region2.centroid))

    if centroid_dist1 > dist_threshold or centroid_dist2 > dist_threshold or daughter_dist > dist_threshold:
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


def check_growth_by_region(cell: cell_class.Cell, region) -> bool:
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