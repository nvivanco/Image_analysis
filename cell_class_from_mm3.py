from __future__ import print_function, division
import numpy as np
from scipy import ndimage as ndi


# this is the object that holds all information for a cell
class Cell:
	# initialize (birth) the cell
	def __init__(self, pxl2um, time_table, cell_id, region, t, parent_id=None):
		"""The cell must be given a unique cell_id and passed the region
		information from the segmentation

		Parameters
		__________

		cell_id : str
			cell_id is a string in the form fXpXtXrX
			f is 3 digit FOV number
			p is 4 digit peak number
			t is 4 digit time point at time of birth
			r is region label for that segmentation
			Use the function create_cell_id to return a proper string.

		region : region properties object
			Information about the labeled region from
			skimage.measure.regionprops()

		parent_id : str
			id of the parent if there is one.
		"""
		# Hack for json deserialization -- there's probably a better way to do this.
		if pxl2um is None:
			return

		# create all the attributes
		# id
		self.id = cell_id

		self.pxl2um = pxl2um
		self.time_table = time_table

		# identification convenience
		self.fov = int(cell_id.split("f")[1].split("p")[0])
		self.peak = int(cell_id.split("p")[1].split("t")[0])
		self.birth_label = int(cell_id.split("r")[1])

		# parent id may be none
		self.parent = parent_id

		# daughters is updated when cell divides
		# if this is none then the cell did not divide
		self.daughters = None

		# birth and division time
		self.birth_time = t
		self.division_time = None  # filled out if cell divides

		# the following information is on a per timepoint basis
		self.times = [t]
		self.abs_times = [time_table[self.fov][t]]  # elapsed time in seconds
		self.labels = [region.label]
		self.bboxes = [region.bbox]
		self.areas = [region.area]

		# calculating cell length and width by using Feret Diamter. These values are in pixels
		length_tmp, width_tmp = feretdiameter(region)
		if length_tmp is None:
			print("feretdiameter() failed for " + self.id + " at t=" + str(t) + ".")
		self.lengths = [length_tmp]
		self.widths = [width_tmp]

		# calculate cell volume as cylinder plus hemispherical ends (sphere). Unit is px^3
		self.volumes = [
			(length_tmp - width_tmp) * np.pi * (width_tmp / 2) ** 2
			+ (4 / 3) * np.pi * (width_tmp / 2) ** 3
		]

		# angle of the fit elipsoid and centroid location
		if region.orientation > 0:
			self.orientations = [-(np.pi / 2 - region.orientation)]
		else:
			self.orientations = [np.pi / 2 + region.orientation]

		self.centroids = [region.centroid]

		# these are special datatype, as they include information from the daugthers for division
		# computed upon division
		self.times_w_div = None
		self.lengths_w_div = None
		self.widths_w_div = None

		# this information is the "production" information that
		# we want to extract at the end. Some of this is for convenience.
		# This is only filled out if a cell divides.
		self.sb = None  # in um
		self.sd = None  # this should be combined lengths of daughters, in um
		self.delta = None
		self.tau = None
		self.elong_rate = None
		self.septum_position = None
		self.width = None

	def grow(self, time_table, region, t):
		"""Append data from a region to this cell.
		use cell.times[-1] to get most current value"""

		self.times.append(t)
		# TODO: Switch time_table to be passed in directly.
		self.abs_times.append(time_table[self.fov][t])
		self.labels.append(region.label)
		self.bboxes.append(region.bbox)
		self.areas.append(region.area)

		# calculating cell length and width by using Feret Diamter
		length_tmp, width_tmp = feretdiameter(region)
		if length_tmp is None:
			print("feretdiameter() failed for " + self.id + " at t=" + str(t) + ".")
		self.lengths.append(length_tmp)
		self.widths.append(width_tmp)
		self.volumes.append(
			(length_tmp - width_tmp) * np.pi * (width_tmp / 2) ** 2
			+ (4 / 3) * np.pi * (width_tmp / 2) ** 3
		)

		if region.orientation > 0:
			ori = -(np.pi / 2 - region.orientation)
		else:
			ori = np.pi / 2 + region.orientation

		self.orientations.append(ori)
		self.centroids.append(region.centroid)

	def divide(self, daughter1, daughter2, t):
		"""Divide the cell and update stats.
		daugther1 and daugther2 are instances of the Cell class.
		daughter1 is the daugther closer to the closed end."""

		# put the daugther ids into the cell
		self.daughters = [daughter1.id, daughter2.id]

		# give this guy a division time
		self.division_time = daughter1.birth_time

		# update times
		self.times_w_div = self.times + [self.division_time]
		self.abs_times.append(self.time_table[self.fov][self.division_time])

		# flesh out the stats for this cell
		# size at birth
		self.sb = self.lengths[0] * self.pxl2um

		# force the division length to be the combined lengths of the daughters
		self.sd = (daughter1.lengths[0] + daughter2.lengths[0]) * self.pxl2um

		# delta is here for convenience
		self.delta = self.sd - self.sb

		# generation time. Use more accurate times and convert to minutes
		self.tau = np.float64((self.abs_times[-1] - self.abs_times[0]) / 60.0)

		# include the data points from the daughters
		self.lengths_w_div = [l * self.pxl2um for l in self.lengths] + [self.sd]
		self.widths_w_div = [w * self.pxl2um for w in self.widths] + [
			((daughter1.widths[0] + daughter2.widths[0]) / 2) * self.pxl2um
		]

		# volumes for all timepoints, in um^3
		self.volumes_w_div = []
		for i in range(len(self.lengths_w_div)):
			self.volumes_w_div.append(
				(self.lengths_w_div[i] - self.widths_w_div[i])
				* np.pi
				* (self.widths_w_div[i] / 2) ** 2
				+ (4 / 3) * np.pi * (self.widths_w_div[i] / 2) ** 3
			)

		# calculate elongation rate.

		try:
			times = np.float64((np.array(self.abs_times) - self.abs_times[0]) / 60.0)
			log_lengths = np.float64(np.log(self.lengths_w_div))
			p = np.polyfit(times, log_lengths, 1)  # this wants float64
			self.elong_rate = p[0] * 60.0  # convert to hours

		except:
			self.elong_rate = np.float64("NaN")
			print("Elongation rate calculate failed for {}.".format(self.id))

		# calculate the septum position as a number between 0 and 1
		# which indicates the size of daughter closer to the closed end
		# compared to the total size
		self.septum_position = daughter1.lengths[0] / (
				daughter1.lengths[0] + daughter2.lengths[0]
		)

		# calculate single width over cell's life
		self.width = np.mean(self.widths_w_div)

		# convert data to smaller floats. No need for float64
		# see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
		convert_to = "float16"  # numpy datatype to convert to

		self.sb = self.sb.astype(convert_to)
		self.sd = self.sd.astype(convert_to)
		self.delta = self.delta.astype(convert_to)
		self.elong_rate = self.elong_rate.astype(convert_to)
		self.tau = self.tau.astype(convert_to)
		self.septum_position = self.septum_position.astype(convert_to)
		self.width = self.width.astype(convert_to)

		self.lengths = [length.astype(convert_to) for length in self.lengths]
		self.lengths_w_div = [
			length.astype(convert_to) for length in self.lengths_w_div
		]
		self.widths = [width.astype(convert_to) for width in self.widths]
		self.widths_w_div = [width.astype(convert_to) for width in self.widths_w_div]
		self.volumes = [vol.astype(convert_to) for vol in self.volumes]
		self.volumes_w_div = [vol.astype(convert_to) for vol in self.volumes_w_div]
		# note the float16 is hardcoded here
		self.orientations = [
			np.float16(orientation) for orientation in self.orientations
		]
		self.centroids = [
			(y.astype(convert_to), x.astype(convert_to)) for y, x in self.centroids
		]

	def place_in_cell(self, x, y, t):
		"""Translates from screen-space to in-cell coordinates."""
		# check if our cell exists at the current timestamp.
		if not (t in self.times):
			return None, None, None

		cell_time = self.times.index(t)
		bbox = self.bboxes[cell_time]
		# check that the point is inside the cell's bounding box.
		if not ((bbox[0] < y) & (y < bbox[2]) & (bbox[1] < x) & (x < bbox[3])):
			return None, None, None

		centroid = self.centroids[cell_time]
		orientation = self.orientations[cell_time]
		dx = x - centroid[1]
		dy = y - centroid[0]
		if orientation < 0:
			orientation = np.pi + orientation
		disp_y = dy * np.sin(orientation) - dx * np.cos(orientation)
		disp_x = dy * np.cos(orientation) + dx * np.sin(orientation)

		return disp_y, disp_x, cell_time

	def print_info(self):
		"""prints information about the cell"""
		print("id = %s" % self.id)
		print("times = {}".format(", ".join("{}".format(t) for t in self.times)))
		print(
			"lengths = {}".format(", ".join("{:.2f}".format(l) for l in self.lengths))
		)

# functions for pruning a dictionary of cells

class Cells(dict):
    def __init__(self, dict_):
        super().__init__(dict_)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# obtains cell length and width of the cell using the feret diameter
def feretdiameter(region):
    """
    feretdiameter calculates the length and width of the binary region shape. The cell orientation
    from the ellipsoid is used to find the major and minor axis of the cell.
    See https://en.wikipedia.org/wiki/Feret_diameter.

    Parameters
    ----------
    region : skimage.measure._regionprops._RegionProperties
        regionprops object of the binary region of the cell

    Returns
    -------
    length : float
        length of the cell
    width : float
        width of the cell
    """

    # y: along vertical axis of the image; x: along horizontal axis of the image;
    # calculate the relative centroid in the bounding box (non-rotated)
    # print(region.centroid)
    y0, x0 = region.centroid
    y0 = y0 - np.int16(region.bbox[0]) + 1
    x0 = x0 - np.int16(region.bbox[1]) + 1

    # orientation is now measured in RC coordinates - quick fix to convert
    # back to xy
    if region.orientation > 0:
        ori1 = -np.pi / 2 + region.orientation
    else:
        ori1 = np.pi / 2 + region.orientation
    cosorient = np.cos(ori1)
    sinorient = np.sin(ori1)

    amp_param = (
        1.2  # amplifying number to make sure the axis is longer than actual cell length
    )

    # coordinates relative to bounding box
    # r_coords = region.coords - [np.int16(region.bbox[0]), np.int16(region.bbox[1])]

    # limit to perimeter coords. pixels are relative to bounding box
    region_binimg = np.pad(
        region.image, 1, "constant"
    )  # pad region binary image by 1 to avoid boundary non-zero pixels
    distance_image = ndi.distance_transform_edt(region_binimg)
    r_coords = np.where(distance_image == 1)
    r_coords = list(zip(r_coords[0], r_coords[1]))

    # coordinates are already sorted by y. partion into top and bottom to search faster later
    # if orientation > 0, L1 is closer to top of image (lower Y coord)
    if (ori1) > 0:
        L1_coords = r_coords[: int(np.round(len(r_coords) / 4))]
        L2_coords = r_coords[int(np.round(len(r_coords) / 4)) :]
    else:
        L1_coords = r_coords[int(np.round(len(r_coords) / 4)) :]
        L2_coords = r_coords[: int(np.round(len(r_coords) / 4))]

    #####################
    # calculte cell length
    L1_pt = np.zeros((2, 1))
    L2_pt = np.zeros((2, 1))

    # define the two end points of the the long axis line
    # one pole.
    L1_pt[1] = x0 + cosorient * 0.5 * region.major_axis_length * amp_param
    L1_pt[0] = y0 - sinorient * 0.5 * region.major_axis_length * amp_param

    # the other pole.
    L2_pt[1] = x0 - cosorient * 0.5 * region.major_axis_length * amp_param
    L2_pt[0] = y0 + sinorient * 0.5 * region.major_axis_length * amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    # aka calcule the closest coordiante in the region to each of the above points.
    # pt_L1 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L1_pt[0],2) + np.power(Pt[1]-L1_pt[1],2)) for Pt in r_coords])]
    # pt_L2 = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-L2_pt[0],2) + np.power(Pt[1]-L2_pt[1],2)) for Pt in r_coords])]

    try:
        pt_L1 = L1_coords[
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - L1_pt[0], 2) + np.power(Pt[1] - L1_pt[1], 2)
                    )
                    for Pt in L1_coords
                ]
            )
        ]
        pt_L2 = L2_coords[
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - L2_pt[0], 2) + np.power(Pt[1] - L2_pt[1], 2)
                    )
                    for Pt in L2_coords
                ]
            )
        ]
        length = np.sqrt(
            np.power(pt_L1[0] - pt_L2[0], 2) + np.power(pt_L1[1] - pt_L2[1], 2)
        )
    except:
        length = None

    #####################
    # calculate cell width
    # draw 2 parallel lines along the short axis line spaced by 0.8*quarter of length = 0.4, to avoid  in midcell

    # limit to points in each half
    W_coords = []
    if (ori1) > 0:
        W_coords.append(
            r_coords[: int(np.round(len(r_coords) / 2))]
        )  # note the /2 here instead of /4
        W_coords.append(r_coords[int(np.round(len(r_coords) / 2)):])
    else:
        W_coords.append(r_coords[int(np.round(len(r_coords) / 2)):])
        W_coords.append(r_coords[: int(np.round(len(r_coords) / 2))])

    # starting points
    x1 = x0 + cosorient * 0.5 * length * 0.4
    y1 = y0 - sinorient * 0.5 * length * 0.4
    x2 = x0 - cosorient * 0.5 * length * 0.4
    y2 = y0 + sinorient * 0.5 * length * 0.4
    W1_pts = np.zeros((2, 2))
    W2_pts = np.zeros((2, 2))

    # now find the ends of the lines
    # one side
    W1_pts[0, 1] = x1 - sinorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[0, 0] = y1 - cosorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[1, 1] = x2 - sinorient * 0.5 * region.minor_axis_length * amp_param
    W1_pts[1, 0] = y2 - cosorient * 0.5 * region.minor_axis_length * amp_param

    # the other side
    W2_pts[0, 1] = x1 + sinorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[0, 0] = y1 + cosorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[1, 1] = x2 + sinorient * 0.5 * region.minor_axis_length * amp_param
    W2_pts[1, 0] = y2 + cosorient * 0.5 * region.minor_axis_length * amp_param

    # calculate the minimal distance between the points at both ends of 3 lines
    pt_W1 = np.zeros((2, 2))
    pt_W2 = np.zeros((2, 2))
    d_W = np.zeros((2, 1))
    i = 0
    for W1_pt, W2_pt in zip(W1_pts, W2_pts):

        # # find the points closest to the guide points
        # pt_W1[i,0], pt_W1[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W1_pt[0],2) + np.power(Pt[1]-W1_pt[1],2)) for Pt in r_coords])]
        # pt_W2[i,0], pt_W2[i,1] = r_coords[np.argmin([np.sqrt(np.power(Pt[0]-W2_pt[0],2) + np.power(Pt[1]-W2_pt[1],2)) for Pt in r_coords])]

        # find the points closest to the guide points
        pt_W1[i, 0], pt_W1[i, 1] = W_coords[i][
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - W1_pt[0], 2) + np.power(Pt[1] - W1_pt[1], 2)
                    )
                    for Pt in W_coords[i]
                ]
            )
        ]
        pt_W2[i, 0], pt_W2[i, 1] = W_coords[i][
            np.argmin(
                [
                    np.sqrt(
                        np.power(Pt[0] - W2_pt[0], 2) + np.power(Pt[1] - W2_pt[1], 2)
                    )
                    for Pt in W_coords[i]
                ]
            )
        ]

        # calculate the actual width
        d_W[i] = np.sqrt(
            np.power(pt_W1[i, 0] - pt_W2[i, 0], 2)
            + np.power(pt_W1[i, 1] - pt_W2[i, 1], 2)
        )
        i += 1

    # take the average of the two at quarter positions
    width = np.mean([d_W[0], d_W[1]])
    return length, width
