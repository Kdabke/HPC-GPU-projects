# cv2 and openslide were needed to install


from collections import OrderedDict
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import color
import re
import subprocess

import openslide


class FileHandler(object):
    def __init__(self):
        """The handler is responsible for storing the processed data, parsing
        the metadata from original file, and reading it from storage. 
        """
        self.metadata = {
            ("available_mag", None),
            ("base_mag", None),
            ("vendor", None),
            ("mpp  ", None),
            ("base_shape", None),
        }
        pass

    def __load_metadata(self):
        raise NotImplementedError

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.
        `read_mpp` is in X, Y format
        """
        raise NotImplementedError

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.
        Args:
            coords (tuple): (dims_x, dims_y), 
                          top left coordinates of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading` 
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading`       
        """
        raise NotImplementedError

    def get_dimensions(self, read_mag=None, read_mpp=None):
        """Will be in X, Y."""
        if read_mpp is not None:
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]
        scale = read_mag / self.metadata["base_mag"]
        # may off some pixels wrt existing mag
        return (self.metadata["base_shape"] * scale).astype(np.int32)

    def prepare_reading(self, read_mag=None, read_mpp=None, cache_path=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.
        `read_mpp` is in X, Y format.
        """
        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        if scale_factor is None:
            self.image_ptr = None
            self.read_lv = read_lv
        else:
            np.save(cache_path, self.get_full_img(read_mag=read_mag))
            self.image_ptr = np.load(cache_path, mmap_mode="r")
        return

    def _get_read_info(self, read_mag=None, read_mpp=None):
        if read_mpp is not None:
            assert read_mpp[0] == read_mpp[1], "Not supported uneven `read_mpp`"
            read_scale = (self.metadata["base_mpp"] / read_mpp)[0]
            read_mag = read_scale * self.metadata["base_mag"]

        hires_mag = read_mag
        scale_factor = None
        if read_mag not in self.metadata["available_mag"]:
            if read_mag > self.metadata["base_mag"]:
                scale_factor = read_mag / self.metadata["base_mag"]
                hires_mag = self.metadata["base_mag"]
            else:
                mag_list = np.array(self.metadata["available_mag"])
                mag_list = np.sort(mag_list)[::-1]
                hires_mag = mag_list - read_mag
                # only use higher mag as base for loading
                hires_mag = hires_mag[hires_mag > 0]
                # use the immediate higher to save compuration
                hires_mag = mag_list[np.argmin(hires_mag)]
                scale_factor = read_mag / hires_mag

        hires_lv = self.metadata["available_mag"].index(hires_mag)
        return hires_lv, scale_factor


class OpenSlideHandler(FileHandler):
    """Class for handling OpenSlide supported whole-slide images."""

    def __init__(self, file_path):
        """file_path (string): path to single whole-slide image."""
        super().__init__()
        self.file_ptr = openslide.OpenSlide(file_path)  # load OpenSlide object
        self.metadata = self.__load_metadata()

        # only used for cases where the read magnification is different from
        self.image_ptr = None  # the existing modes of the read file
        self.read_lv = None

    def __load_metadata(self):
        metadata = {}

        wsi_properties = self.file_ptr.properties
        level_0_magnification = wsi_properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        level_0_magnification = float(level_0_magnification)

        downsample_level = self.file_ptr.level_downsamples
        magnification_level = [level_0_magnification / lv for lv in downsample_level]

        mpp = [
            wsi_properties[openslide.PROPERTY_NAME_MPP_X],
            wsi_properties[openslide.PROPERTY_NAME_MPP_Y],
        ]
        mpp = np.array(mpp)

        metadata = [
            ("available_mag", magnification_level),  # highest to lowest mag
            ("base_mag", magnification_level[0]),
            ("vendor", wsi_properties[openslide.PROPERTY_NAME_VENDOR]),
            ("mpp  ", mpp),
            ("base_shape", np.array(self.file_ptr.dimensions)),
        ]
        return OrderedDict(metadata)

    def read_region(self, coords, size):
        """Must call `prepare_reading` before hand.
        Args:
            coords (tuple): (dims_x, dims_y), 
                          top left coordinates of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading` 
            size (tuple): (dims_x, dims_y)
                          width and height of image region at selected 
                          `read_mag` or `read_mpp` from `prepare_reading`       
        """
        if self.image_ptr is None:
            # convert coord from read lv to lv zero
            lv_0_shape = np.array(self.file_ptr.level_dimensions[0])
            lv_r_shape = np.array(self.file_ptr.level_dimensions[self.read_lv])
            up_sample = (lv_0_shape / lv_r_shape)[0]
            new_coord = [0, 0]
            new_coord[0] = int(coords[0] * up_sample)
            new_coord[1] = int(coords[1] * up_sample)
            region = self.file_ptr.read_region(new_coord, self.read_lv, size)
        else:
            region = self.image_ptr[
                coords[1] : coords[1] + size[1], coords[0] : coords[0] + size[0]
            ]
        return np.array(region)[..., :3]

    def get_full_img(self, read_mag=None, read_mpp=None):
        """Only use `read_mag` or `read_mpp`, not both, prioritize `read_mpp`.
        `read_mpp` is in X, Y format.
        """

        read_lv, scale_factor = self._get_read_info(
            read_mag=read_mag, read_mpp=read_mpp
        )

        read_size = self.file_ptr.level_dimensions[read_lv]

        wsi_img = self.file_ptr.read_region((0, 0), read_lv, read_size)
        wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
        if scale_factor is not None:
            # now rescale then return
            if scale_factor > 1.0:
                interp = cv2.INTER_CUBIC
            else:
                interp = cv2.INTER_LINEAR
            wsi_img = cv2.resize(
                wsi_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
            )
        return wsi_img


def get_file_handler(path, backend):
    if backend in [
            '.svs', '.tif', 
            '.vms', '.vmu', '.ndpi',
            '.scn', '.mrxs', '.tiff',
            '.svslide',
            '.bif',
            ]:
        return OpenSlideHandler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend


import glob
import inspect
import logging
import os
import shutil

import cv2
import numpy as np
from scipy import ndimage


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.
    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


####
def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


####
def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


####
def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.
    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


####
def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


####
def log_debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


####
def log_info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.
    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 
    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size
    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

import cv2
import math
import random
import colorsys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm


####
def colorize(ch, vmin, vmax):
    """Will clamp value value outside the provided range to vmax and vmin."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


####
def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


####
def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.
    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


####
def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.
    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} , 
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


####
def gen_figure(
    imgs_list,
    titles,
    fig_inch,
    shape=None,
    share_ax="all",
    show=False,
    colormap=plt.get_cmap("jet"),
):
    """Generate figure."""
    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig


