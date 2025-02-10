import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon

# Helper functions for image loading and setup

def _create_figure_and_axes(phase_stack, num_images):
    """Creates the Matplotlib figure and axes."""
    figxsize = phase_stack.shape[2] * num_images / 100.0
    figysize = phase_stack.shape[1] / 100.0
    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(figxsize, figysize),
                            facecolor="white", edgecolor="black",
                            gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'top': 1, 'bottom': 0})
    return fig, axs.flatten()

def _create_color_dict(data_source):
    """Creates the color dictionary based on the data source."""
    if isinstance(data_source, np.ndarray):  # For mask_path
        num_colors = max(20, len(np.unique(data_source)))
    elif isinstance(data_source, dict):  # For cells_dict
        num_colors = max(20, len(data_source))
    elif isinstance(data_source, pd.DataFrame):  # For cells_df
        num_colors = max(20, data_source['cell_id'].nunique())  # Use nunique for efficiency
    else:
        return {}  # Return empty if no data source

    cmap = plt.cm.get_cmap('tab20', num_colors)
    if isinstance(data_source, np.ndarray):  # For mask_path
        color_dict = {j: cmap(j)[:3] for j in range(num_colors + 1)}
    elif isinstance(data_source, dict):  # For cells_dict
        color_dict = {cell_id: cmap(i)[:3] for i, cell_id in enumerate(data_source)}
    elif isinstance(data_source, pd.DataFrame):  # For cells_df
        color_dict = {cell_id: cmap(color_i)[:3] for color_i, cell_id in enumerate(data_source['cell_id'].unique())}

    return color_dict


# Helper functions for plotting

def _plot_mask(ax, mask, color_dict, alpha):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            colored_mask[x, y] = color_dict[mask[x, y]]
    ax.imshow(colored_mask, alpha=alpha)

def _plot_cell_contour(ax, phase, region, color):
    cell_mask = np.zeros(phase.shape, dtype=int)
    rr, cc = polygon(np.array(region.coords)[:, 0], np.array(region.coords)[:, 1], phase.shape)
    cell_mask[rr, cc] = 1
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

def _show_and_close_plot(fig):
    plt.show(fig)
    plt.close(fig)


# Main plotting functions

def display_segmentation(path_to_original_stack, mask_path, alpha=0.5, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    mask_stack = tifffile.imread(mask_path)  # Load mask stack here
    num_images = end - start
    fig, axs = _create_figure_and_axes(phase_stack, num_images)
    color_dict = _create_color_dict(mask_stack[end])  # Create color dict from mask data

    for i in range(start, end):
        phase = phase_stack[i]
        mask = mask_stack[i]
        axs[i - start].imshow(phase, cmap='gray')
        _plot_mask(axs[i-start], mask, color_dict, alpha)
        _set_axes_properties(axs[i-start], i)

    _add_legend(color_dict)
    _show_and_close_plot(fig)


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
    _show_and_close_plot(fig)


def display_stack(path_to_original_stack, start=0, end=20):
    phase_stack = tifffile.imread(path_to_original_stack)
    num_images = end - start
    fig, axs = _create_figure_and_axes(phase_stack, num_images)

    for i in range(start, end):
        phase = phase_stack[i]
        axs[i - start].imshow(phase, cmap='gray')
        _set_axes_properties(axs[i - start], i)

    _show_and_close_plot(fig)


def _create_kymograph(phase_stack, start, end, fov_id, peak_id, output_dir):
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
