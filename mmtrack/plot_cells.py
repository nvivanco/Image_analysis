import os

import tifffile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon
from skimage.measure import find_contours

# Helper functions for image loading and setup

def _create_figure_and_axes(phase_stack, num_images):
    """Creates the Matplotlib figure and axes."""
    
    # Calculate figure size based on image dimensions
    figxsize = phase_stack.shape[2] * num_images / 100.0
    figysize = phase_stack.shape[1] / 100.0
    
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=num_images, 
        figsize=(figxsize, figysize),
        facecolor="white", 
        edgecolor="black",
        gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'top': 1, 'bottom': 0}
    )
    

    if num_images == 1:
        # If there's only one subplot, axs is a single Axes object. Wrap it in an array.
        axs_array = np.array([axs])
    else: 
        # If there are multiple subplots, axs is already a 1D or 2D NumPy array. Flatten it.
        axs_array = axs.flatten()
        
    return fig, axs_array


def _create_color_dict(data_source):
    """Creates the color dictionary based on the data source."""
    
    # 1. Determine unique IDs and number of colors needed
    if isinstance(data_source, np.ndarray):  # For mask_path (segmentation masks)
        # FIX: Get the actual unique labels from the mask array.
        # This is essential to prevent KeyError when labels are not sequential (e.g., 0, 1, 5, 37).
        unique_labels = np.unique(data_source)
        # Filter out label 0 (background) for coloring, unless it's the only one.
        labels_to_color = unique_labels[unique_labels != 0] if len(unique_labels) > 1 else unique_labels
        
        num_colors = max(20, len(labels_to_color))

    elif isinstance(data_source, dict):  # For cells_dict
        labels_to_color = data_source.keys()
        num_colors = max(20, len(labels_to_color))
        
    elif isinstance(data_source, pd.DataFrame):  # For cells_df
        # Get unique cell_ids to use as keys
        labels_to_color = data_source['cell_id'].unique()
        num_colors = max(20, len(labels_to_color))
        
    else:
        return {}  # Return empty if no valid data source

    # 2. Select Color Map
    # Use a cyclical color map to handle many labels
    cmap = plt.colormaps['tab20'](np.linspace(0, 1, num_colors))

    # 3. Create the Dictionary
    color_dict = {}
    
    # Map each unique label to a unique color from the color map
    for i, label in enumerate(labels_to_color):
        # Use the modulo operator to cycle through the available colors (e.g., if there are >20 cells)
        color_dict[label] = cmap[i % len(cmap), :3]

    # Special handling for mask background (label 0)
    if isinstance(data_source, np.ndarray) and 0 in unique_labels:
        # Assign a transparent or neutral color for label 0 (often black [0, 0, 0] or white [1, 1, 1])
        # Since the `_plot_mask` function uses the color_dict, we need to provide a value for 0.
        # It's safest to give it a visible (but usually black/white) color here; the alpha blending will handle visibility.
        color_dict[0] = [0.0, 0.0, 0.0] # Black for background if label 0 exists

    return color_dict

def mask_from_region(region, image):
    cell_mask = np.zeros(image.shape, dtype=int)
    rr, cc = polygon(np.array(region.coords)[:, 0], np.array(region.coords)[:, 1], image.shape)
    cell_mask[rr, cc] = 1
    return cell_mask

def plot_kymograph_cells_id(phase_kymograph, fluor_kymograph, full_region_df, folder, fov_id, peak_id, track_id_col='track_id'):
    fig, ax = plt.subplots(1,1, figsize=(40, 10))

    # Get kymograph shape once for both calls
    kymograph_shape = phase_kymograph.shape

    ax.imshow(phase_kymograph, cmap = 'grey')
    _plot_cell_masks(ax, full_region_df, kymograph_shape, y_coord_col = 'centroid_y', x_coord_col = 'centroid_x', lineage_col = track_id_col)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Phase Kymograph - {folder} FOV: {fov_id}, trench: {peak_id}')

    plt.xlabel("Time frames")
    plt.tight_layout()

    plt.savefig(f'{folder}_FOV_{fov_id}_trench_{peak_id}_kymograph.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_cell_masks(ax, full_region_df, kymograph_shape, y_coord_col='centroid-0', x_coord_col='centroid-1',
                     lineage_col=None):
    default_cell_contour_color = '#AA5486'
    default_centroid_color = '#FC8F54'

    # Prepare data for LineCollection for contours
    all_contours_segments = []
    all_contour_colors = []

    # Prepare data for centroids
    centroid_x_coords = []
    centroid_y_coords = []
    centroid_colors = []

    if lineage_col:
        unique_track_ids = full_region_df[lineage_col].dropna().unique()
        colors_cmap = plt.get_cmap('tab20', len(unique_track_ids))
        track_colors = {track_id: colors_cmap(i) for i, track_id in enumerate(unique_track_ids)}

    for index, region_props in full_region_df.iterrows():
        # 'coords' are assumed to be (row, col) pixels within the mask
        cell_pixel_coords = np.array(region_props['coords'])  # e.g., [[r1,c1], [r2,c2], ...]

        # Calculate bounding box for the current cell's mask
        min_row, min_col = np.min(cell_pixel_coords, axis=0)
        max_row, max_col = np.max(cell_pixel_coords, axis=0)

        # Create a small temporary mask for the current cell
        # Add a small buffer to ensure contours are fully captured if they go to edge
        buffer = 1
        bbox_min_row = max(0, min_row - buffer)
        bbox_min_col = max(0, min_col - buffer)
        bbox_max_row = min(kymograph_shape[0], max_row + buffer)
        bbox_max_col = min(kymograph_shape[1], max_col + buffer)

        temp_mask_shape = (bbox_max_row - bbox_min_row + 1, bbox_max_col - bbox_min_col + 1)
        temp_mask = np.zeros(temp_mask_shape, dtype=np.uint8)

        # Map cell_pixel_coords to relative coordinates within temp_mask
        relative_rows = cell_pixel_coords[:, 0] - bbox_min_row
        relative_cols = cell_pixel_coords[:, 1] - bbox_min_col

        # Populate the temporary mask
        temp_mask[relative_rows, relative_cols] = 1

        # Find contours on this small temporary mask
        # level=0.5 means it finds contours at the boundary between 0 and 1
        # fully_connected='high' means it considers 8-connectivity for background, 4-connectivity for foreground
        contours = find_contours(temp_mask, level=0.5, fully_connected='high')

        if not contours:
            continue  # Skip if no contour found (e.g., single pixel or degenerate mask)

        # `find_contours` returns (row, col) coordinates for the contour.
        # We need to convert them back to global kymograph coordinates.
        # And convert to (x, y) for plotting (col, row)
        global_contours = []
        for contour in contours:
            # Shift back to global coordinates and swap for (x, y) plotting
            global_contour_x = contour[:, 1] + bbox_min_col
            global_contour_y = contour[:, 0] + bbox_min_row
            global_contours.append(np.vstack([global_contour_x, global_contour_y]).T)

        y_coord = region_props[y_coord_col]
        x_coord = region_props[x_coord_col]

        # Determine color for the current cell
        if lineage_col and region_props[lineage_col] in track_colors:
            current_color = track_colors[region_props[lineage_col]]
        else:
            current_color = default_cell_contour_color

        # Add all contours for this cell to the main list, with the determined color
        for contour_segment in global_contours:
            all_contours_segments.append(contour_segment)
            all_contour_colors.append(current_color)

        # Add centroid data
        centroid_x_coords.append(x_coord)
        centroid_y_coords.append(y_coord)
        centroid_colors.append(current_color if lineage_col else default_centroid_color)

    # Plot all cell contours at once using LineCollection
    if all_contours_segments:  # Only plot if there are segments to draw
        line_collection = LineCollection(all_contours_segments, colors=all_contour_colors, linewidths=0.5)
        ax.add_collection(line_collection)

    # Plot all centroids at once using scatter
    if centroid_x_coords:  # Only plot if there are centroids
        ax.scatter(centroid_x_coords, centroid_y_coords, color=centroid_colors, s=5, zorder=2)

# Helper functions for plotting

def _plot_mask(ax, mask, color_dict, alpha):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            colored_mask[x, y] = color_dict[mask[x, y]]
    ax.imshow(colored_mask, alpha=alpha)

def _plot_cell_contour(ax, phase, region, color):
    cell_mask = mask_from_region(region, phase)
    ax.contour(cell_mask, levels=[0.5], colors=[color], linewidths=1)


# --- Helper functions for plot finishing ---
def _set_axes_properties(ax, i):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(f"{i}", fontsize=8)

def _add_legend(color_dict):
    legend_patches = [patches.Patch(color=color, label=f"Label {label}") for label, color in color_dict.items()]
    if legend_patches:
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')


# Main plotting functions

def display_segmentation(path_to_original_stack, mask_path, alpha=0.5, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    mask_stack = tifffile.imread(mask_path)  
    
    # 1. Determine the number of images to display
    num_images = end - start
    
    # 2. Check for bounds and adjust if necessary
    # Ensure 'end' doesn't exceed the stack length. 
    # If the user-provided 'end' is out of bounds, use the stack length.
    if end > mask_stack.shape[0]:
        end = mask_stack.shape[0]
        num_images = end - start
    
    # 3. Create the figure/axes
    fig, axs = _create_figure_and_axes(phase_stack, num_images)
    
    # CRITICAL FIX: Create color dict from the entire time range being plotted.
    # This ensures every unique cell label that appears in frames 'start' to 'end' 
    # is included in the dictionary, preventing the KeyError.
    mask_subset = mask_stack[start:end]
    color_dict = _create_color_dict(mask_subset) 

    # 4. Loop and plot each frame
    for i in range(start, end):
        phase = phase_stack[i]
        mask = mask_stack[i]
        
        # Determine the correct axis index (since the loop starts at 'start')
        ax_index = i - start 
        
        axs[ax_index].imshow(phase, cmap='gray')
        _plot_mask(axs[ax_index], mask, color_dict, alpha)
        _set_axes_properties(axs[ax_index], i)

    _add_legend(color_dict)
    
    # 5. Display the plot 
    plt.show()

def display_cells_from_df(path_to_original_stack, cells_df, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    num_images = end - start
    fig, axs = _create_figure_and_axes(phase_stack, num_images)
    color_dict = _create_color_dict(cells_df)

    for i in range(start, end):
        phase = phase_stack[i]
        axs[i - start].imshow(phase, cmap='gray')
        for cell_id in cells_df['cell_id'].unique():
            if i in cells_df['time_index'].to_numpy():
                row = cells_df[(cells_df['cell_id'] == cell_id) & (cells_df['time_index'] == i)].index[0]
                cell_mask = cells_df.loc[row, 'masks']
                axs[i - start].contour(cell_mask, levels=[0.5], colors= [color_dict[cell_id]], linewidths=1)
                y_coord, x_coord = cells_df.loc[row, 'centroids']
                axs[i - start].scatter(x_coord, y_coord, color=color_dict[cell_id], s=5)

        _set_axes_properties(axs[i - start], i)

    _add_legend(color_dict)
    return fig

def display_cells_from_dict(path_to_original_stack, cells_dict, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    num_images = end - start
    fig, axs = _create_figure_and_axes(phase_stack, num_images)
    color_dict = _create_color_dict(cells_dict)

    for i in range(start, end):
        phase = phase_stack[i]
        axs[i - start].imshow(phase, cmap='gray')
        for cell_id, cell in cells_dict.items():
            if i in cell.times:
                n = cell.times.index(i)
                region = cell.regions[n]
                _plot_cell_contour(axs[i - start], phase, region, color_dict[cell_id])
                centroid = cell.centroids[n]
                y_coord, x_coord = centroid
                axs[i - start].scatter(x_coord,y_coord, color = color_dict[cell_id], s = 5)

        _set_axes_properties(axs[i - start], i)

    _add_legend(color_dict)
    return fig


def display_stack(path_to_original_stack, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    num_images = end - start
    fig, axs = _create_figure_and_axes(phase_stack, num_images)

    for i in range(start, end):
        phase = phase_stack[i]
        axs[i - start].imshow(phase, cmap='gray')
        _set_axes_properties(axs[i - start], i)

    plt.show()


def create_kymograph(phase_stack, start, end, fov_id, peak_id, output_dir):
    """Creates and saves a kymograph."""
    kymographs_gray = []
    for i in range(start, end):
        phase = phase_stack[i]
        if phase.ndim == 3:
            kymographs_gray.append(np.mean(phase, axis=2))
        else:
            kymographs_gray.append(phase)

    combined_kymograph = np.concatenate(kymographs_gray, axis=1)
    lin_filename = f'{fov_id}_{peak_id}.tif'
    lin_filepath = os.path.join(output_dir, lin_filename)  # Use output_dir
    tifffile.imwrite(lin_filepath, combined_kymograph)
    return combined_kymograph

# kymograph processing
def add_time_frame_df(full_region_df, labeled_stack_px_width, mask_kymograph_px_width, x_centroid_col='centroid-1'):
    time_frame_pixel_dict = _make_time_frame_pixel_dict(labeled_stack_px_width, mask_kymograph_px_width)
    full_region_df['time_frame'] = full_region_df['centroid-1'].apply(_map_pixel_to_index,
                                                                      range_dict=time_frame_pixel_dict)

def _make_time_frame_pixel_dict(labeled_stack_px_width, mask_kymograph_px_width):
    enumerated_x_increments_dict = {index: value for index, value in enumerate(
        list(range(labeled_stack_px_width, mask_kymograph_px_width + 1, labeled_stack_px_width)))}
    time_frame_pixel_dict = {}
    for time_frame, x_limit in enumerated_x_increments_dict.items():
        time_frame_pixel_dict[time_frame] = (x_limit - labeled_stack_px_width, x_limit - 1)
    return time_frame_pixel_dict

def _map_pixel_to_index(pixel, range_dict):
    for index, (min_val, max_val) in range_dict.items():
        if min_val <= pixel <= max_val:
            return index
    return None