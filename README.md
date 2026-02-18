# Geo_AI

A platform for applying AI to solve geographical problems.

## Hackathon Objectives

1.  Derive a DTM from point cloud data.
2.  Estimate natural surface water flow patterns.
3.  Design drainage plans for the region using the derived DTM.

## Deliverables

1.  Automated AI/ML pipelines to derive a DTM from point cloud data.
2.  Drainage network provided as Shapefiles.
3.  Documentation and final report.

## Folder Structure

    README.md        -- This file
    ./doc            -- All documentation-related files
    ./src            -- All source code
    ./src/DTM_xx.xx  -- Code for DTM generation
    ./src/flows_xx.xx-- Code for natural flow estimation
    ./src/drain_xx.xx-- Code for optimal drainage design
    ./src/misc_xx.xx -- Miscellaneous scripts and utilities
    ./src/config.py  -- Configurations for the project
    ./data           -- All data files
    ./data/input     -- Input datasets
    ./data/output    -- Output datasets
    ./data/models    -- Models and trained weights
    ./data/vectors   -- Vector data (Shapefiles, etc.)   

All code will contain a `config.py` file with the necessary
configurations to control algorithm parameters.
