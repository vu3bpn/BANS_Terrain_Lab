#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:32:18 2026

@author: bipin
"""


import os
import random
import numpy as np
import geopandas
import pandas
from sklearn.neighbors import NearestNeighbors, KDTree
from config import *
from misc_utilities import *
from DTM_test_nn import *


class StreamingDTMDataset(IterableDataset):
    def __init__(self,batch_size = 64,n_batches = 32,seq_len = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.seq_len = seq_len
        self.files_list = []
        self.selected_cols = ["X","Y","Z"]  
        self.read_files()
        self.file_idx = 0


    def read_files(self):
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
                gdf_id = row1[1]['id']
                geometry = row1[1].geometry                
                input_file_path = os.path.join(debug_dir,f"dtm_debug_{las_file_name}_id_{gdf_id}.las")
                self.files_list.append((input_file_path,geometry,gdf_id))

    def __len__(self):
        return self.n_batches
    
    @lru_cache(maxsize=5)
    def get_df_tree(self,file_tuple):
        las_file_path, geometry,gdf_id = file_tuple
        dtm_csv_filename = os.path.join(debug_csv_dir,f"DTM_df_{gdf_id}.csv")
        
        if os.path.exists(dtm_csv_filename):
            dtm_df = pandas.read_csv(dtm_csv_filename)
        else:            
            dtm_file = laspy.read(las_file_path)
            dtm_record = dtm_file.points
            dtm_header = dtm_file.header       
            
            dtm_df = pandas.DataFrame(dtm_record.array)
            dtm_df['X'] = dtm_df['X']*dtm_header.scales[0] + dtm_header.offsets[0]
            dtm_df['Y'] = dtm_df['Y']*dtm_header.scales[1] + dtm_header.offsets[1]
            dtm_df['Z'] = dtm_df['Z']*dtm_header.scales[2] + dtm_header.offsets[2]     
            dtm_df.to_csv(dtm_csv_filename)
            
        xy = dtm_df[["X","Y"]]
        tree = KDTree(xy,metric='l1')
        return dtm_df,tree

    def get_dataset(self):
        #file_tuple = random.choice(self.files_list)
        file_tuple =  self.files_list[self.file_idx]
        
        
        dtm_df,tree = self.get_df_tree(file_tuple)
        center_idx = random.sample(range(len(dtm_df)),1)[0]
        center_point = dtm_df.iloc[center_idx][["X","Y"]]
        dist, tree_idx = tree.query([center_point], k=self.seq_len)
        tree_idx = tree_idx.flatten()        
       
        dtm_data =  dtm_df.iloc[tree_idx]           
        input_dataset = np.array(dtm_data[self.selected_cols])
        return input_dataset
       
        

    def __iter__(self):
        batches = range(self.n_batches)  
        for batch in batches:
            x = self.get_dataset()
            yield x
            


if __name__ == "__main__":
    '''test data'''
    dtm_data_stream = StreamingDTMDataset()
    for xyz in dtm_data_stream:
        tree = KDTree(xyz)
        
        
        