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

def knn_fill(df,k= 5,measure='mean'):
    '''fill non terrain (classification != 2 ) to using nearest knn values'''
    ground_df = df.query("classification==2")
    knn = NearestNeighbors(n_neighbors=k)
    ground_points = np.array(ground_df[["X","Y","Z"]])
    knn.fit(ground_points)
    def get_z(row):
        if row['classification'] == 2:
            return row['Z']
        index = knn.kneighbors(np.array([row[["X","Y","Z"]]]),
                                    return_distance=False)
        nearest_ground_vectors = ground_points[index]
        #zvals = [x[2] for x in nearest_ground_vectors]
        z_vals = nearest_ground_vectors[:,:,2]
        if measure == 'min':
            min_z = np.min(z_vals)
            return min_z
        mean_z = np.mean(z_vals,dtype='int32')
        return mean_z
    df["Z"] =  df.apply(get_z, axis=1)
    return df



if __name__ == "__main1__":
    test_df = pandas.DataFrame({"X":range(50),"Y":range(50),"Z":range(50),"classification":2*np.ones(50)})
    test_df['Y'] += 50
    test_df['Z'] += 150
    classification = list(test_df['classification'])
    classification[20:30] = [0]*10
    test_df['classification'] = classification #test_df['classification']*(test_df['X']!=25)
    filled_df = knn_fill(test_df)
    
    

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
            
            dtm_debug_file = os.path.join(debug_dir,f"dtm_debug_{las_file_name}_id_{id}.las")
            dsm_debug_file = os.path.join(debug_dir,f"dsm_debug_{las_file_name}_id_{id}.las")
            terrain_debug_file = os.path.join(debug_dir,f"terrain_debug_{las_file_name}_id_{id}.las")
            
            '''
            dtm_debug_csv_file = os.path.join(debug_dir,f"dtm_debug_{las_file_name}_id_{id}.csv")
            dsm_debug_csv_file = os.path.join(debug_dir,f"dsm_debug_{las_file_name}_id_{id}.csv")
            terrain_debug_csv_file = os.path.join(debug_dir,f"terrain_debug_{las_file_name}_id_{id}.csv")
            '''
            
            geometry = row1[1].geometry
            dtm_record,dtm_header = subset_with_geom(dtm_file_path,geometry)            
            df = pandas.DataFrame(dtm_record.array)            
            df_to_las(df,dtm_header,terrain_debug_file)
            #df.to_csv(terrain_debug_csv_file)
            
            
            filled_df = knn_fill(df,k=50,measure='min')            
            df_to_las(filled_df,dtm_header,dtm_debug_file)
            #filled_df.to_csv(dtm_debug_csv_file)

            dsm_record,dsm_header = subset_with_geom(las_file_path,geometry)
            dsm_df = pandas.DataFrame(dsm_record.array)
            #dsm_df.to_csv(dsm_debug_csv_file)
            #df_to_las(dsm_df,dsm_header,dsm_debug_file)
            
            with laspy.open(dsm_debug_file, mode='w', header=dsm_header) as writer:
                writer.write_points(dsm_record)


            #df_to_las(df,header,terrain_debug_file)
            #df_to_las(dsm_df,header,dsm_debug_file)
            
            
            
            #filled_df.to_csv(dtm_debug_csv_file)
            #df.to_csv(terrain_debug_file)
            #dsm_df.to_csv(dsm_debug_csv_file)
            
            

