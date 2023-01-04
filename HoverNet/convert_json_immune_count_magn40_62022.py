import sys
import os

import json
json_path_wsi = str(sys.argv[1])
sample_name = str(sys.argv[1])
sample_name2 = sample_name.split('.', 1)[0]

centroid_list_wsi = []
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
        inst_type = inst_info['type']
        type_list_wsi.append(inst_type)

import pandas as pd
df_centroid = pd.DataFrame(centroid_list_wsi, columns =['coords_1', 'coords_2'])
df_type = pd.DataFrame(type_list_wsi, columns = ['celltype'])
df_combined = pd.concat([df_centroid, df_type], axis = 1)
df_combined['Sample'] = pd.Series([sample_name2 for x in range(len(df_combined.index))])

df_combined.round(0)

df_combined['celltype'] = df_combined['celltype'].replace(
    to_replace=[0,1,2,3,4,5],
    value=['nolabe', 'neopla', 'inflam', 'connec', 'necros', 'no-neo'])

df_combined['is_immune'] = df_combined['celltype'].replace(
    to_replace=['nolabe', 'neopla', 'inflam', 'connec', 'necros', 'no-neo'],
    value=['non_immune', 'non_immune', 'immune', 'non_immune', 'non_immune', 'non_immune'])

extension = '.centroid.csv'
out = sample_name2+extension

df_combined.to_csv(out, sep='\t', index=False)
