import sys
import os
#sys.executable

#%matplotlib inline
#from matplotlib import pyplot as plt
#from matplotlib import rcParams
#rcParams['figure.facecolor'] = (1,1,1,1)
#rcParams['svg.fonttype'] = 'none'
# commented lines above should be read in only if interested in overlay visualization of images


import seaborn as sns
import numpy as np
import pandas as pd
import json

## only read in these functions when overlay is required
#from functions_hovernet.py import *

# load the json file (may take ~20 secs)


json_path_wsi = str(sys.argv[1])
######### change the file name here

bbox_list_wsi = []
centroid_list_wsi = []
contour_list_wsi = [] 
type_list_wsi = []

# add results to individual lists
with open(json_path_wsi) as json_file:
    data = json.load(json_file)
    mag_info = data['mag']
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        centroid_list_wsi.append(inst_centroid)
        inst_contour = inst_info['contour']
        contour_list_wsi.append(inst_contour)
        inst_bbox = inst_info['bbox']
        bbox_list_wsi.append(inst_bbox)
        inst_type = inst_info['type']
        type_list_wsi.append(inst_type)


# define the region to select
# multiply each um with 4.0733 (which is 2(magnification)/pixel width of image (0.491))
##### change the numbers here for specific regions of the image from Qupath 
x_tile = 0
y_tile = 0
w_tile = int(sys.argv[2])
h_tile = int(sys.argv[2])

## only run this when overlay image output is needed to visualize the tile being analyzed
#wsi_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))

# only consider results that are within the tile

coords_xmin = x_tile
coords_xmax = x_tile + w_tile
coords_ymin = y_tile
coords_ymax = y_tile + h_tile

tile_info_dict = {}
count = 0
for idx, cnt in enumerate(contour_list_wsi):
    cnt_tmp = np.array(cnt)
    cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
    label = str(type_list_wsi[idx])
    if cnt_tmp.shape[0] > 0:
        cnt_adj = np.round(cnt_tmp - np.array([x_tile,y_tile])).astype('int')
        tile_info_dict[idx] = {'contour': cnt_adj, 'type':label}
        count += 1

# getting out the types of cells in the defined tile of image

cell_type_list = []
for key in tile_info_dict.keys():
    cell_type_list.append((tile_info_dict.get(key).get("type")))

out =  sys.argv[3] + '.txt'

textfile = open(str(out), "w") ######### change output file name based on the json file being analyzed
for element in cell_type_list:
    textfile.write(element + "\n")
textfile.close()
