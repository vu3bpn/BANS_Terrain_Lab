#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:41:43 2026

@author: bipin
"""

import pdal
import json
from pathlib import Path
from config import *

if __name__ == "__main__":
    las_input_filenames = list(Path(input_laz_dir).rglob("*.laz"))  
    for las_file in las_input_filenames:
        output_filename = os.path.join(copc_dir, las_file.name )
        pipeline = {
            "pipeline": [
                str(las_file),
                {
                    "type": "writers.copc",
                    "filename": output_filename
                }
            ]
        }    
        p = pdal.Pipeline(json.dumps(pipeline))
        p.execute()