#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 16:05:24 2026

@author: bipin
"""

import pdal
import json
import numpy as np
from pathlib import Path
import os
import laspy
from config import *
import rasterio
import pandas
from multiprocessing import Pool


ground_pipeline = {
    "pipeline": [
        {
            "type": "readers.las",
        },
        {
                "type": "filters.smrf"
          },
        {
        "type":"writers.las",
        }
    ]
}
    
    
def run_dtm_pipeline(las_file):    
    las_filename = str(las_file.name)
    dtm_filename = f"{os.path.splitext(las_filename)[0]}_dtm.las"
    dtm_output_path = os.path.join(dtm_dir,dtm_filename)
    if os.path.exists(dtm_output_path):
        log(f"Skipping DTM file {dtm_output_path}\n")
        return None
    
    ground_pipeline["pipeline"][0]["filename"] = str(las_file)
    ground_pipeline["pipeline"][-1]["filename"] = dtm_output_path  
    #ground_pipeline["pipeline"][3]["a_srs"] = f"EPSG:{input_epsg}"
    pipeline_json = json.dumps(ground_pipeline)
    print(pipeline_json)
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    
    #%% Apply crs to generated file
    '''
    input_crs_wkt = laspy.read(las_file).header.parse_crs().to_wkt()
    with rasterio.open(dtm_output_path,"r+") as dtm_file:
        dtm_file.crs = rasterio.crs.CRS.from_wkt(input_crs_wkt)
    log(f"generated DTM : {dtm_output_path}")
    '''
    
    
if __name__ == "__main1__":  
    '''dtm pipeline sequential'''
    las_input_filenames = list(Path(input_laz_dir).rglob("*.las")) +  list(Path(input_laz_dir).rglob("*.LAS")) 
    for filename1 in las_input_filenames:
        run_dtm_pipeline(filename1)
    
if __name__ == "__main__":  
    '''dtm pipeline parallel'''
    las_input_filenames = list(Path(input_laz_dir).rglob("*.las"))  +  list(Path(input_laz_dir).rglob("*.LAS"))  
    with Pool(pipeline_cuncurrent_jobs) as p:
        p.map(run_dtm_pipeline,las_input_filenames)

if __name__ == "__main1__":    
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))  
    for las_file in las_input_filenames:    
        print(las_file)
        las_file_p = laspy.open(las_file)
        variables = list(las_file_p.header.point_format.dimension_names)
        csv_filename_prefix = os.path.split(las_file)[-1].strip(".laz")
        for i, points in enumerate(las_file_p.chunk_iterator(chunk_size)):
            print(f"Chunk {i}, size: {len(points.x)}")
            points_csv_filename  = os.path.join(debug_csv_dir,csv_filename_prefix+f"_chunk-{i}.csv")
            points_dict = {x:list(points[x]) for x in variables}
            points_df = pandas.DataFrame(points_dict)
            points_df.to_csv(points_csv_filename)
        
        
        
        