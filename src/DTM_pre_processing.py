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
from multiprocessing import Pool

if __name__ == "__main__":
    ground_pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": "input.las"
            },
            {
                # Classify ground points automatically
                "type": "filters.smrf",
                #"type": "filters.pmf",
                "slope": 0.15,
                "window": 18.0,
                "threshold": 0.5,
                "scalar": 1.25
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"
            },
            {
                "type": "writers.gdal",
                "filename": "dtm.tif",
                "resolution": dtm_resolution,
                "output_type": "min",
                "gdaldriver": "GTiff",
                "gdalopts": "COMPRESS=LZW"
            }
        ]
    }
    
    
def run_dtm_pipeline(las_file):    
    las_filename = str(las_file.name)
    dtm_filename = f"{os.path.splitext(las_filename)[0]}_dtm.tif"
    ground_pipeline["pipeline"][0]["filename"] = str(las_file)
    dtm_output_path = os.path.join(dtm_dir,dtm_filename)
    ground_pipeline["pipeline"][3]["filename"] = dtm_output_path  
    #ground_pipeline["pipeline"][3]["a_srs"] = f"EPSG:{input_epsg}"
    pipeline = pdal.Pipeline(json.dumps(ground_pipeline))
    pipeline.execute()
    
    #%% Apply crs to generated file
    input_crs_wkt = laspy.read(las_file).header.parse_crs().to_wkt()
    with rasterio.open(dtm_output_path,"r+") as dtm_file:
        dtm_file.crs = rasterio.crs.CRS.from_wkt(input_crs_wkt)
    log(f"generated DTM : {dtm_output_path}")
    
    
if __name__ == "__main__":    
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))   
    with Pool(pipeline_cuncurrent_jobs) as p:
        p.map(run_dtm_pipeline,las_input_filenames)

if __name__ == "__main1__":
    for las_file in las_input_filenames:        
        #input_las_path = os.path.join(input_laz_dir, las_file)
        run_dtm_pipeline(las_file)
        
        
        
        