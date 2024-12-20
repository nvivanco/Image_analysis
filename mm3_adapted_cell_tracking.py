import os
from typing import Tuple
from skimage import segmentation, morphology
from skimage.measure import regionprops
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import six
import tifffile
import cell_class_from_mm3 as cell

def find_cell_intensities(path_to_original_stack,
                          path_to_segmented_stack,
                          cells,
                          midline=False):
    """
    Finds fluorescenct information for cells. All the cells in cells
    should be from one fov/peak. See the function
    organize_cells_by_channel()
    """

    # Load fluorescent images and segmented images for this microfluidic channel

    fl_stack = tifffile.imread(path_to_original_stack)
    seg_stack = tifffile.imread(path_to_segmented_stack)

    t0 = 0

    # Loop through cells
    for cell in cells.values():
        # give this cell two lists to hold new information
        cell.fl_tots = []  # total fluorescence per time point
        cell.fl_area_avgs = []  # avg fluorescence per unit area by timepoint
        cell.fl_vol_avgs = []  # avg fluorescence per unit volume by timepoint

        if midline:
            cell.mid_fl = []  # avg fluorescence of midline

        # and the time points that make up this cell's life
        for n, t in enumerate(cell.times):
            # create fluorescent image only for this cell and timepoint.
            fl_image_masked = np.copy(fl_stack[t - t0])
            fl_image_masked[seg_stack[t - t0] != cell.labels[n]] = 0

            # append total flourescent image
            cell.fl_tots.append(np.sum(fl_image_masked))
            # and the average fluorescence
            cell.fl_area_avgs.append(np.sum(fl_image_masked) / cell.areas[n])
            cell.fl_vol_avgs.append(np.sum(fl_image_masked) / cell.volumes[n])

            if midline:
                # add the midline average by first applying morphology transform
                bin_mask = np.copy(seg_stack[t - t0])
                bin_mask[bin_mask != cell.labels[n]] = 0
                med_mask, _ = morphology.medial_axis(bin_mask, return_distance=True)
                # med_mask[med_dist < np.floor(cap_radius/2)] = 0
                # print(img_fluo[med_mask])
                if np.shape(fl_image_masked[med_mask])[0] > 0:
                    cell.mid_fl.append(np.nanmean(fl_image_masked[med_mask]))
                else:
                    cell.mid_fl.append(0)

    # The cell objects in the original dictionary will be updated,
    # no need to return anything specifically.

def make_lineage_chnl_stack(path_to_stack: str,
                            labeled_stack: str,
                            fov_id: int,
                            peak_id: int,
                            time_btwn_frames: int,
                            pxl2um: float) -> dict:
    """
    Fromm mm3 track.py
    Create the lineage for a set of segmented images for one channel. Start by making the regions in the first time points potenial cells.
    Go forward in time and map regions in the timepoint to the potential cells in previous time points, building the life of a cell.
    Used basic checks such as the regions should overlap, and grow by a little and not shrink too much. If regions do not link back in time, discard them.
    If two regions map to one previous region, check if it is a sensible division event.

    Parameters
    ----------
    fov_id: int
    peak_id: int
    time_btwn_frames: int (in minutes)
    pxl2um: float, pixels to microns conversion

    Returns
    -------
    cells : dict
        A dictionary of all the cells from this lineage, divided and undivided

    """

    # load in parameters
    # if leaf regions see no action for longer than this, drop them
    lost_cell_time = 8
    # only cells with y positions below this value will recieve the honor of becoming new
    # cells, unless they are daughters of current cells
    new_cell_y_cutoff = 250
    # only regions with labels less than or equal to this value will be considered to start cells
    new_cell_region_cutoff = 15


    # start time is the first time point for this series of TIFFs.
    start_time_index = 0

    image_data_seg = tifffile.imread(labeled_stack)


    # Calculate all data for all time points.
    # this list will be length of the number of time points
    regions_by_time = [regionprops(label_image=timepoint) for timepoint in image_data_seg]
    time_table = {}
    time_table[int(fov_id)] = {}
    for time in range(image_data_seg.shape[0]):
        t_in_seconds = np.around(
                            (time - 0) * 60*time_btwn_frames, #seconds per time index
                            decimals=0,
                        ).astype("uint32")
        time_table[int(fov_id)][time] = t_in_seconds

    # Set up data structures.
    cells: dict[str, cell.Cell] = {}  # Dict that holds all the cell objects, divided and undivided
    cell_leaves: list = []  # cell ids of the current leaves of the growing lineage tree

    # go through regions by timepoint and build lineages
    # timepoints start with the index of the first image
    for t, regions in enumerate(regions_by_time, start=start_time_index):
        # if there are cell leaves who are still waiting to be linked, but
        # too much time has passed, remove them.
        cell_leaves, cells = prune_leaves(cell_leaves, cells, lost_cell_time, t)

        # make all the regions leaves if there are no current leaves
        if not cell_leaves:
            for region in regions:
                cell_leaves, cells = add_leaf_orphan(
                    region,
                    cell_leaves,
                    cells,
                    new_cell_y_cutoff,
                    new_cell_region_cutoff,
                    t,
                    peak_id,
                    fov_id,
                    pxl2um,
                    time_table)

        # Determine if the regions are children of current leaves
        else:
            cell_leaves, cells = make_leaf_region_map(
                regions,
                pxl2um,
                time_table,
                cell_leaves,
                cells,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                t,
                peak_id,
                fov_id)

    ## plot kymograph with lineage overlay & save it out
    make_lineage_plot(path_to_stack,
    labeled_stack,
    fov_id,
    peak_id,
    cells)

    # return the dictionary with all the cells
    return cells


def make_lineage_plot(path_to_stack: str,
    labeled_stack: str,
    fov_id: int,
    peak_id: int,
    cells: dict[str, cell.Cell]):
    """Produces a lineage image for the first valid FOV containing cells

    Parameters
    ----------
    plot_out_dir: str
        path to save image lineage map
    fov_id: int
        current FOV
    peak_id: int
        current peak (trap)
    cells: dict[str, Cell]
        dict of Cell objects

    Returns
    -------
    None
    """
    #time offset from time_table
    start_time_index= 0
    # plotting lineage trees for complete cells
    general_dir = os.path.dirname(labeled_stack)
    plot_out_dir = os.path.join(general_dir, 'lineage_plot')
    os.makedirs(plot_out_dir, exist_ok=True)

    fig, ax = plot_lineage_images(
        cells,
        fov_id,
        peak_id,
        path_to_stack,
        labeled_stack,
        t_adj=start_time_index,
    )
    lin_filename = f'{fov_id}_{peak_id}.tif'

    lin_filepath = os.path.join(plot_out_dir, lin_filename)
    try:
        fig.savefig(lin_filepath, dpi=75)
    # sometimes image size is too large for matplotlib renderer
    except ValueError:
        warning("Image size may be too large for matplotlib")
    plt.close(fig)

def plot_lineage_images(
    cells,
    fov_id,
    peak_id,
    path_to_stack,
    labeled_stack,
    t_adj=1):
    """
    Plot linages over images across time points for one FOV/peak.
    Parameters
    ----------
    bgcolor : Designation of background to use. Subtracted images look best if you have them.
    fgcolor : Designation of foreground to use. This should be a segmented image.
    t_adj : int
        adjust time indexing for differences between t index of image and image number
    """

    # filter cells
    cells = find_cells_of_fov_and_peak(cells, fov_id, peak_id)

    # load subtracted and segmented data
    image_data_bg = tifffile.imread(path_to_stack)
    image_data_seg = tifffile.imread(labeled_stack)

    fig, ax = plot_cells(image_data_bg, image_data_seg, True, t_adj)

    # Annotate each cell with information
    fig, ax = plot_tracks(cells, t_adj, fig, ax)

    return fig, ax

def find_cells_of_fov_and_peak(cells, fov_id, peak_id) -> cell.Cells:
    """Return only cells from a specific fov/peak
    Parameters
    ----------
    fov_id : int corresponding to FOV
    peak_id : int correstonging to peak
    """

    fcells = {}  # f is for filtered

    for cell_id in cells:
        if cells[cell_id].fov == fov_id and cells[cell_id].peak == peak_id:
            fcells[cell_id] = cells[cell_id]

    return fcells

def plot_regions(
    seg_data: np.ndarray,
    regions: list,
    ax: plt.Axes,
    cmap: str,
    vmin: float = 0.5,
    vmax: float = 100) -> plt.Axes:
    """
    Plot segmented cells from one peak & time step

    Parameters
    ----------
    seg_data: np.ndarray
        segmentation labels
    regions: list
        list of regionprops objects
    ax: plt.Axes
        current axis

    Returns
    -------
    ax: plt.Axes
        updated axis
    """
    # make a new version of the segmented image where the
    # regions are relabeled by their y centroid position.
    # scale it so it falls within 100.
    seg_relabeled = seg_data.copy().astype(float)
    for region in regions:
        rescaled_color_index = region.centroid[0] / seg_data.shape[0] * vmax
        seg_relabeled[seg_relabeled == region.label] = (
            int(rescaled_color_index) - 0.1
        )  # subtract small value to make it so there is not overlabeling
    ax.imshow(seg_relabeled, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    return ax

def plot_tracks(
    cells: dict[str, cell.Cell], t_adj: int, fig: plt.Figure, ax: plt.Axes
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking tracked cells across time

    Parameters
    ----------
    cells: dict[str, Cell]
        dictionary of Cell objects
    t_adj: int
        time offset
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object

    Returns
    -------
    fig: plt.Figure
        updated figure object
    ax: plt.Axes
        updated axis object
    """
    for cell_id in cells:
        for n, t in enumerate(cells[cell_id].times):
            t -= t_adj  # adjust for special indexing

            x = cells[cell_id].centroids[n][1]
            y = cells[cell_id].centroids[n][0]

            # add a circle at the centroid for every point in this cell's life
            circle = mpatches.Circle(
                xy=(x, y), radius=2, color="white", lw=0, alpha=0.5
            )
            ax[t].add_patch(circle)

            # draw connecting lines between the centroids of cells in same lineage
            try:
                if n < len(cells[cell_id].times) - 1:
                    fig, ax = connect_centroids(
                        fig, ax, cells[cell_id], n, t, t_adj, x, y
                    )
            except:
                pass

            # draw connecting between mother and daughters
            try:
                if n == len(cells[cell_id].times) - 1 and cells[cell_id].daughters:
                    fig, ax = connect_mother_daughter(
                        fig, ax, cells, cell_id, t, t_adj, x, y
                    )
            except:
                pass
    return fig, ax

def connect_centroids(
    fig: plt.Figure,
    ax: plt.Axes,
    cell: cell.Cell,
    n: int,
    t: int,
    t_adj: int,
    x: int,
    y: int,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking cell centroids in time

    Parameters
    ----------
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object
    cell: Cell
        current cell object
    n: int
        cell time index relative to birth
    t: int
        time step (relative to experiment start)
    x: int
        centroid x position
    y: int
        centroid y position

    Returns:
    fig: plt.Figure
        updated figure
    ax: plt.Axes
        updated axis
    """
    # coordinates of the next centroid
    x_next = cell.centroids[n + 1][1]
    y_next = cell.centroids[n + 1][0]
    t_next = cell.times[n + 1] - t_adj  # adjust for special indexing

    transFigure = fig.transFigure.inverted()

    # get coordinates for the whole figure
    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
    coord2 = transFigure.transform(ax[t_next].transData.transform([x_next, y_next]))

    # create line
    line = mpl.lines.Line2D(
        (coord1[0], coord2[0]),
        (coord1[1], coord2[1]),
        transform=fig.transFigure,
        color="white",
        lw=1,
        alpha=0.5,
    )

    # add it to plot
    fig.lines.append(line)
    return fig, ax


def connect_mother_daughter(
    fig: plt.Figure,
    ax: plt.Axes,
    cells: dict[str, cell.Cell],
    cell_id: str,
    t: int,
    t_adj: int,
    x: int,
    y: int,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw lines linking mother to its daughters

    Parameters
    ----------
    fig: plt.Figure
        figure object
    ax: plt.Axes
        axis object
    cells: dict[str, Cell]
        dictionary of Cell objects
    cell_id: str
        current mother cell id
    t: int
        time step (relative to experiment start)
    t_adj: int
        time offset from time_table
    x: int
        centroid x position
    y: int
        centroid y position

    Returns
    -------
    fig: plt.Figure
        updated figure
    ax: plt.Axes
        updated axis
    """
    # daughter ids
    d1_id = cells[cell_id].daughters[0]
    d2_id = cells[cell_id].daughters[1]

    # both daughters should have been born at the same time.
    t_next = cells[d1_id].times[0] - t_adj

    # coordinates of the two daughters
    x_d1 = cells[d1_id].centroids[0][1]
    y_d1 = cells[d1_id].centroids[0][0]
    x_d2 = cells[d2_id].centroids[0][1]
    y_d2 = cells[d2_id].centroids[0][0]

    transFigure = fig.transFigure.inverted()

    # get coordinates for the whole figure
    coord1 = transFigure.transform(ax[t].transData.transform([x, y]))
    coordd1 = transFigure.transform(ax[t_next].transData.transform([x_d1, y_d1]))
    coordd2 = transFigure.transform(ax[t_next].transData.transform([x_d2, y_d2]))

    # create line and add it to plot for both
    for coord in [coordd1, coordd2]:
        line = mpl.lines.Line2D(
            (coord1[0], coord[0]),
            (coord1[1], coord[1]),
            transform=fig.transFigure,
            color="white",
            lw=1,
            alpha=0.5,
            ls="dashed",
        )
        # add it to plot
        fig.lines.append(line)
    return fig, ax

def plot_cells(
    image_data_bg: np.ndarray, image_data_seg: np.ndarray, fgcolor: bool, t_adj: int
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot phase imaging data overlaid with segmented cells

    Parameters
    ----------
    image_data_bg: np.ndarray
        phase contrast images
    image_data_seg: np.ndarray
        segmented images
    fgcolor: bool
        whether to plot segmented images
    t_adj: int
        time offset from time_table

    Returns
    -------
    fig: plt.Figure
        matplotlib figure
    ax: plt.Axes
        matplotlib axis
    """
    n_imgs = image_data_bg.shape[0]
    image_indices = range(n_imgs)

    # Trying to get the image size down
    figxsize = image_data_bg.shape[2] * n_imgs / 100.0
    figysize = image_data_bg.shape[1] / 100.0

    # plot the images in a series
    fig, axes = plt.subplots(
        ncols=n_imgs,
        nrows=1,
        figsize=(figxsize, figysize),
        facecolor="black",
        edgecolor="black",
    )
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # change settings for each axis
    ax = axes.flat  # same as axes.ravel()
    for a in ax:
        a.set_axis_off()
        a.set_aspect("equal")
        ttl = a.title
        ttl.set_position([0.5, 0.05])

    for i in image_indices:
        ax[i].imshow(image_data_bg[i], cmap=plt.cm.gray, aspect="equal")

        if fgcolor:
            # calculate the regions across the segmented images
            regions_by_time = [regionprops(timepoint) for timepoint in image_data_seg]
            # Color map for good label colors
            cmap = mpl.colors.ListedColormap(
                sns.husl_palette(n_colors=100, h=0.5, l=0.8, s=1)
            )
            cmap.set_under(color="black")
            ax[i] = plot_regions(image_data_seg[i], regions_by_time[i], ax[i], cmap)

        ax[i].set_title(str(i + t_adj), color="white")

    return fig, ax

def prune_leaves(
    cell_leaves: list[str], cells: dict[str, cell.Cell], lost_cell_time: int, t: int) -> Tuple[list, dict]:
    """
    Remove leaves for cells that have been lost for more than lost_cell_time

    Parameters
    ----------
    cell_leaves: list[str]
        list of current cell leaves
    cells: dict[str, Cell]
        dictionary of all Cell objects
    lost_cell_time: int
        number of time steps after which to drop lost cells
    t: int
        current time step

    Returns
    -------
    cell_leaves: list[str]
        updated list of cell leaves
    cells: dict[str,Cell]
        updated dictionary of Cell objects
    """
    for leaf_id in cell_leaves:
        if t - cells[leaf_id].times[-1] > lost_cell_time:
            cell_leaves.remove(leaf_id)
    return cell_leaves, cells


def handle_two_regions(
		region1,
		region2,
		cells: dict[str, cell.Cell],
		cell_leaves: list[str],
		pxl2um: float,
		leaf_id: str,
		t: int,
		peak_id: int,
		fov_id: int,
		time_table: dict,
		y_cutoff: int,
		region_cutoff: int):
	"""
	Classify the two regions as either a divided cell (two daughters), or one growing cell and one trash.

	Parameters
	----------
	region1: RegionProps object
		first region
	region2: RegionProps object
		second region
	cells: dict
		dictionary of Cell objects
	params: dict
		parameter dictionary
	leaf_id: int
		id of current tracked cell
	t: int
		time step
	peak_id: int
		peak (trap) number
	fov_id: int
		current fov
	time_table: dict
		dictionary of time points
	y_cutoff: int
		y position threshold for new cells
	region_cutoff: int
		region label cutoff for new cells

	Returns
	-------
	cell_leaves: list[str]
		list of cell leaves
	cells: dict
		updated dicitonary of"""

	# check_division returns 3 if cell divided,
	# 1 if first region is just the cell growing and the second is trash
	# 2 if the second region is the cell, and the first is trash
	# or 0 if it cannot be determined.
	check_division_result = check_division(cells[leaf_id], region1, region2)

	if check_division_result == 3:

		daughter1_id, daughter2_id, cells = divide_cell(
			region1, region2, t, peak_id, fov_id, pxl2um, cells, time_table, leaf_id)
		# remove mother from current leaves
		cell_leaves.remove(leaf_id)

		# add the daughter ids to list of current leaves if they pass cutoffs
		cell_leaves = add_leaf_daughter(
			region1, cell_leaves, daughter1_id, y_cutoff, region_cutoff
		)

		cell_leaves = add_leaf_daughter(
			region2, cell_leaves, daughter2_id, y_cutoff, region_cutoff
		)

	# 1 means that daughter 1 is just a continuation of the mother
	# The other region should be a leaf it passes the requirements
	elif check_division_result == 1:
		cells[leaf_id].grow(time_table, region1, t)

		cell_leaves, cells = add_leaf_orphan(
			region2,
			cell_leaves,
			cells,
			y_cutoff,
			region_cutoff,
			t,
			peak_id,
			fov_id,
			pxl2um,
			time_table)

	# ditto for 2
	elif check_division_result == 2:
		cells[leaf_id].grow(time_table, region2, t)

		cell_leaves, cells = add_leaf_orphan(
			region1,
			cell_leaves,
			cells,
			y_cutoff,
			region_cutoff,
			t,
			peak_id,
			fov_id,
			pxl2um,
			time_table)

	return cell_leaves, cells

def add_leaf_daughter(
    region, cell_leaves: list[str], id: str, y_cutoff: int, region_cutoff: int) -> list:
    """
    Add new leaf to tree if it clears thresholds

    Parameters
    ----------
    region: RegionProps object
        candidate region
    cell_leaves: list[str]
        list of cell leaves
    id: str
        candidate cell id
    y_cutoff: int
        max distance from closed end of channel to allow new cells
    region_cutoff: int
        max region (labeled ascending from closed end of channel)

    Returns
    -------
    cell_leaves: list[str]
        updated list of cell leaves
    """
    # add the daughter ids to list of current leaves if they pass cutoffs
    if region.centroid[0] < y_cutoff and region.label <= region_cutoff:
        cell_leaves.append(id)

    return cell_leaves


def handle_discarded_regions(
    cell_leaves: list[str],
    region_links: list,
    regions,
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    cells: dict[str, cell.Cell],
    pxl2um: float,
    time_table: dict,
    t: int,
    peak_id: int, fov_id: int) -> Tuple[list, dict]:
    """
    Process third+ regions down from closed end of channel. They will either be discarded or made into new cells.

    Parameters
    ----------
    cell_leaves: list[str]
        list of current cell leaves
    region_links: list
        regions from current time step, sorted by y position
    regions: list
        list of RegionProps objects
    new_cell_y_cutoff: int
        y position cutoff for new cells
    new_cell_region_cutoff: int
        region label cutoff for new cells
    cells: dict[str, Cell]
        dictionary of Cell objects
    pxl2um: float
        conversion factor from pixels to microns
    time_table: dict
        dict of time points
    t: int
        current time step
    peak_id: int
        current channel (trap) id
    fov_id: int
        current fov

    Returns
    -------
    cell_leaves: list[str]
        updated list of leaves by cell id
    cells: dict[str, Cell]
        updated dict of Cell objects
    """
    discarded_regions = sorted(region_links, key=lambda x: x[1])[2:]
    for discarded_region in discarded_regions:
        region = regions[discarded_region[0]]
        if (
            region.centroid[0] < new_cell_y_cutoff
            and region.label <= new_cell_region_cutoff
        ):
            cell_id = create_cell_id(region, t, peak_id, fov_id)
            cells[cell_id] = cell.Cell(
                pxl2um,
                time_table,
                cell_id,
                region,
                t,
                parent_id=None,
            )
            cell_leaves.append(cell_id)  # add to leaves
        else:
            # since the regions are ordered, none of the remaining will pass
            break
    return cell_leaves, cells

def divide_cell(
    region1,
    region2,
    t: int,
    peak_id: int,
    fov_id: int,
    pxl2um: float,
    cells: dict[str, cell.Cell],
    time_table: dict,
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
    time_table: dict
        dictionary of time points
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
        time_table,
        daughter1_id,
        region1,
        t,
        parent_id=leaf_id,
    )
    cells[daughter2_id] = cell.Cell(
        pxl2um,
        time_table,
        daughter2_id,
        region2,
        t,
        parent_id=leaf_id,
    )
    cells[leaf_id].divide(cells[daughter1_id], cells[daughter2_id], t)

    return daughter1_id, daughter2_id, cells

def update_leaf_regions(
    regions: list,
    current_leaf_positions: list[tuple[str, float]],
    leaf_region_map: dict[str, list[tuple[int, float]]]) -> dict[str, list[tuple[int, float]]]:
    """
    Loop through regions from current time step and match them to existing leaves

    Parameters
    ----------
    regions: list
        list of RegionProps objects
    current_leaf_positions: list[tuple[leaf_id, position]]
        list of (leaf_id, cell centroid) for current leaves
    leaf_region_map: dict[str,list[tuple[int,float]]]
        dict whose keys are leaves (cell IDs) and list of values (region number, region location)

    Returns
    ------
    leaf_region_map: dict[str,Tuple[int,int]]
        updated leaf region map
    """
    # go through regions, they will come off in Y position order
    for r, region in enumerate(regions):
        # create tuple which is cell_id of closest leaf, distance
        current_closest = (str(''), float("inf"))

        # check this region against all positions of all current leaf regions,
        # find the closest one in y.
        for leaf in current_leaf_positions:
            # calculate distance between region and leaf
            y_dist_region_to_leaf: float = abs(region.centroid[0] - leaf[1])

            # if the distance is closer than before, update
            if y_dist_region_to_leaf < current_closest[1]:
                current_closest = (leaf[0], y_dist_region_to_leaf)

        # update map with the closest region
        leaf_region_map[current_closest[0]].append((r, y_dist_region_to_leaf))

    return leaf_region_map

def add_leaf_orphan(
    region,
    cell_leaves: list[str],
    cells: dict[str, cell.Cell],
    y_cutoff: int,
    region_cutoff: int,
    t: int,
    peak_id: int,
    fov_id: int,
    pxl2um: float,
    time_table: dict) -> Tuple[list, dict]:
    """
    Add new leaf if it clears thresholds.

    Parameters
    ----------
    region: regionprops object
        candidation region for new cell
    cell_leaves: list[str]
        list of current leaves
    cells: dict[str, Cell]
        dict of Cell objects
    y_cutoff: int
        max distance from closed end of channel to allow new cells
    region_cutoff: int
        max region (labeled ascending from closed end of channel)
    t: int
        time
    peak: int
        peak id
    fov: int
        fov id
    pxl2um: float
        pixel to micron conversion
    time_table: dict
        dictionary of time points

    Returns
    -------
    cell_leaves: list[str]
        updated leaves
    cells: dict[str, Cell]
        updated dict of Cell objects

    """
    if region.centroid[0] < y_cutoff and region.label <= region_cutoff:
        cell_id = create_cell_id(region, t, peak_id, fov_id)
        cells[cell_id] = cell.Cell(
            pxl2um,
            time_table,
            cell_id,
            region,
            t,
            parent_id=None)
        cell_leaves.append(cell_id)  # add to leaves
    return cell_leaves, cells

# see if a cell has reasonably divided
def check_division(cell: cell.Cell, region1, region2) -> int:
    """Checks to see if it makes sense to divide a
    cell into two new cells based on two regions.

    Return 0 if nothing should happend and regions ignored
    Return 1 if cell should grow by region 1
    Return 2 if cell should grow by region 2
    Return 3 if cell should divide into the regions."""

    # load in parameters
    max_growth_length = 2 # ratio
    min_growth_length = 0.9 # ratio

    # see if either region just could be continued growth,
    # if that is the case then just return
    # these shouldn't return true if the cells are divided
    # as they would be too small
    if check_growth_by_region(cell, region1):
        return 1

    if check_growth_by_region(cell, region2):
        return 2

    # make sure combined size of daughters is not too big
    combined_size = region1.major_axis_length + region2.major_axis_length
    # check if length is not too much longer
    if cell.lengths[-1] * max_growth_length < combined_size:
        return 0
    # and not too small
    if cell.lengths[-1] * min_growth_length > combined_size:
        return 0

    # centroids of regions should be in the upper and lower half of the
    # of the mother's bounding box, respectively
    # top region within top half of mother bounding box
    if (
        cell.bboxes[-1][0] > region1.centroid[0]
        or cell.centroids[-1][0] < region1.centroid[0]
    ):
        return 0
    # bottom region with bottom half of mother bounding box
    if (
        cell.centroids[-1][0] > region2.centroid[0]
        or cell.bboxes[-1][2] < region2.centroid[0]
    ):
        return 0

    # if you got this far then divide the mother
    return 3

# take info and make string for cell id
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

def make_leaf_region_map(
    regions: list,
    pxl2um: float,
    time_table: dict,
    cell_leaves: list[str],
    cells: dict[str, cell.Cell],
    new_cell_y_cutoff: int,
    new_cell_region_cutoff: int,
    t: int,
    peak_id: int,
    fov_id: int) -> tuple[list, dict[str, cell.Cell]]:
    """
    Map regions in current time point onto previously tracked cells

    Parameters
    ----------
    regions: list
        regions from current time point
    time_table: dict
        dictionary of time points

    Returns
    -------
    cell_leaves: list[str]
        list of tree leaves
    cells: dict
        dictionary of cell objects
    """
    ### create mapping between regions and leaves
    leaf_region_map: dict[str, list[tuple[int,float]]] = {leaf_id: [] for leaf_id in cell_leaves}

    # get the last y position of current leaves and create tuple with the id
    current_leaf_positions = [
        (leaf_id, cells[leaf_id].centroids[-1][0]) for leaf_id in cell_leaves
    ]

    leaf_region_map = update_leaf_regions(
        regions, current_leaf_positions, leaf_region_map
    )

    # go through the current leaf regions.
    # limit by the closest two current regions if there are three regions to the leaf
    for leaf_id, region_links in six.iteritems(leaf_region_map):
        if len(region_links) > 2:

            leaf_region_map[leaf_id] = get_two_closest_regions(region_links)

            # for the discarded regions, put them as new leaves
            # if they are near the closed end of the channel
            cell_leaves, cells = handle_discarded_regions(
                cell_leaves,
                region_links,
                regions,
                new_cell_y_cutoff,
                new_cell_region_cutoff,
                cells,
                pxl2um,
                time_table,
                t,
                peak_id,
                fov_id)

    cell_leaves, cells = update_region_links(
        cell_leaves,
        cells,
        leaf_region_map,
        regions,
        pxl2um,
        time_table,
        t,
        peak_id,
        fov_id,
        new_cell_y_cutoff,
        new_cell_region_cutoff)

    return cell_leaves, cells

def get_two_closest_regions(region_links: list) -> list:
    """
    Retrieve two regions closest to closed end of the channel.

    Parameters
    ----------
    region_links: list
        list of all linked regions

    Returns
    -------
    closest two regions: list
        two regions closest to closed end of channel.
    """
    closest_two_regions = sorted(region_links, key=lambda x: x[1])[:2]
    # but sort by region order so top region is first
    closest_two_regions = sorted(closest_two_regions, key=lambda x: x[0])
    return closest_two_regions


def update_region_links(
		cell_leaves: list[str],
		cells: dict[str, cell.Cell],
		leaf_region_map: dict[str, list[tuple[int, float]]],
		regions: list,
		pxl2um: float,
		time_table: dict,
		t: int,
		peak_id: int,
		fov_id: int,
		y_cutoff: int,
		region_cutoff: int):
	"""
	Loop over current leaves and connect them to descendants

	Parameters
	----------
	cell_leaves: list[str]
		currently tracked cell_ids
	cells: dict[str, Cell]
		dictionary of Cell objects
	leaf_region_map: dict[str,Tuple[int,float]]
		dictionary with keys = cell id, values = (region number, region centroid)
	regions: list
		list of RegionProps objects
	pxl2um: float
		pixel to uM conversion factor
	time_table: dict
		dictionary of time points
	t: int
		current time step
	peak_id: int
		current peak (trap) id
	fov_id: int
		current fov id

	Returns
	-------
	cell_leaves: list[str]
		list of current leaves labeled by cell id
	cells: dict[str, Cell]
		updated dictionary of Cell objects
	"""

	### iterate over the leaves, looking to see what regions connect to them.
	for leaf_id, region_links in six.iteritems(leaf_region_map):

		# if there is just one suggested descendant,
		# see if it checks out and append the data
		if len(region_links) == 1:
			region = regions[region_links[0][0]]  # grab the region from the list

			# check if the pairing makes sense based on size and position
			# this function returns true if things are okay
			if check_growth_by_region(cells[leaf_id], region):
				# grow the cell by the region in this case
				cells[leaf_id].grow(time_table, region, t)

		# there may be two daughters, or maybe there is just one child and a new cell
		elif len(region_links) == 2:
			# grab these two daughters
			region1 = regions[region_links[0][0]]
			region2 = regions[region_links[1][0]]
			cell_leaves, cells = handle_two_regions(
				region1,
				region2,
				cells,
				cell_leaves,
				pxl2um,
				leaf_id,
				t,
				peak_id,
				fov_id,
				time_table,
				y_cutoff,
				region_cutoff)

	return cell_leaves, cells


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


def cell_growth_func(t, sb, elong_rate):
	"""
	Assumes you have taken log of the data.
	It also allows the size at birth to be a free parameter, rather than fixed
	at the actual size at birth (but still uses that as a guess)
	Assumes natural log, not base 2 (though I think that makes less sense)

	old form: sb*2**(alpha*t)
	"""
	return sb + elong_rate * t

