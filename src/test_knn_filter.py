#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:11:56 2026

@author: bipin
"""
import pandas 
import geopandas
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from config import *
from misc_utilities import *


def knn_fill(df):
    ground_df = df.query("classification==2")
    knn = NearestNeighbors(n_neighbors=4)
    ground_points = np.array(ground_df[["X","Y","Z"]])
    knn.fit(ground_points)
    def get_z(row):
        if row['classification'] == 2:
            return row['Z']
        dist,index = knn.kneighbors(np.array([row[["X","Y","Z"]]]))
        nearest_ground_vectors = ground_points[index]
        mean_z = np.mean(nearest_ground_vectors[:,2],dtype='int32')
        return mean_z
    df["Z"] =  df.apply(get_z, axis=1)
    return df

def df_to_las(df,header,out_file_path):
    total_points = len(df)
    record = laspy.ScaleAwarePointRecord.zeros(total_points, header=header)
    out_header = laspy.LasHeader(point_format=header.point_format,version="1.4")
    out_header.x_scale = header.x_scale
    out_header.y_scale = header.y_scale
    out_header.z_scale = header.z_scale
    out_header.offsets = header.offsets            
    out_crs = header.parse_crs()
    out_header.add_crs(out_crs)
    for var1 in df.columns:
        record[var1] = df[var1].values
    with laspy.open(out_file_path, mode='w', header=out_header) as writer:
                writer.write_points(record)




if __name__ == "__main__":
    data_info_df = pandas.read_csv(data_info_file)
    filename_paths = {os.path.split(x)[-1]:x for x in data_info_df['filename']}
    for las_file in las_vect_dict:
        vect_file = las_vect_dict[las_file]
        shapefile_path = os.path.join(vector_dir,vect_file)
        las_file_path = filename_paths[las_file]
        las_file_name = os.path.splitext(las_file)[0]
        dtm_file = las_file_name+"_dtm.las"
        dtm_file_path = os.path.join(dtm_dir,dtm_file)
        gdf = geopandas.read_file(shapefile_path)
        for row1 in gdf.iterrows():
            id = row1[1]['id']
            geometry = row1[1].geometry
            dtm_record,header = subset_with_geom(dtm_file_path,geometry)
            dsm_record,dsm_header = subset_with_geom(las_file_path,geometry)
            df = pandas.DataFrame(dtm_record.array)
            filled_df = knn_fill(df)
            
            dtm_debug_file = os.path.join(debug_dir,f"dtm_debug_{las_file_name}_id_{id}.las")
            dsm_debug_file = os.path.join(debug_dir,f"dsm_debug_{las_file_name}_id_{id}.las")
            terrain_debug_file = os.path.join(debug_dir,f"terrain_debug_{las_file_name}_id_{id}.las")
            df_to_las(filled_df,dtm_header,dtm_debug_file)
            df_to_las(df,dtm_header,dsm_debug_file)
            

