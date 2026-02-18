import os
from pathlib import Path

#%% configurations for data and input/output directories
data_dir = os.path.join(Path.cwd().parent,"data")
input_dir = os.path.join(data_dir,"input")
input_laz_dir = os.path.join(input_dir,"laz")


#%% Input files
data_url_list = os.path.join(input_dir,"data_links.txt")