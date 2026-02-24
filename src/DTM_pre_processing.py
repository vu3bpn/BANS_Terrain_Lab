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
                  "type": "filters.reprojection",
                  "out_srs": "EPSG:4326"
                },
            {
                "type": "writers.gdal",
                "filename": "dtm.tif",
                "resolution": 1.0,
                "output_type": "min",
                "gdaldriver": "GTiff"
            }
        ]
    }
    
    dtm_pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": "input.las"
                },
                {
                    # Keep only ground points (Classification == 2)
                    "type": "filters.range",
                    "limits": "Classification[2:2]"
                },
                {
                  "type": "filters.reprojection",
                  "out_srs": "EPSG:4326"
                },
                {
                    # Interpolate ground points into a raster DTM
                    "type": "writers.gdal",
                    "filename": "dtm.tif",
                    "resolution": 1.0,        # 1 meter per pixel — adjust as needed
                    "output_type": "min",     # use minimum Z in each cell (bare earth)
                    "gdaldriver": "GTiff"
                }
            ]
        }
    
if __name__ == "__main__":
    
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))    
    

    for las_file in las_input_filenames:        
        #input_las_path = os.path.join(input_laz_dir, las_file)
        input_epsg = laspy.read(las_file).header.parse_crs().to_epsg()
        las_filename = str(las_input_filenames[0].name)
        ground_pipeline["pipeline"][0]["filename"] = str(las_file)
        dtm_output_path = os.path.join(dtm_dir, f"{os.path.splitext(las_filename)[0]}_dtm.tif")
        ground_pipeline["pipeline"][4]["filename"] = dtm_output_path  
        #ground_pipeline["pipeline"][3]["a_srs"] = f"EPSG:{input_epsg}"
    
        pipeline = pdal.Pipeline(json.dumps(ground_pipeline))
        pipeline.execute()
