import os
from pathlib import Path

#%% configurations for data and input/output directories
data_dir = os.path.join(Path.cwd().parent,"data")
input_dir = os.path.join(data_dir,"input")
input_laz_dir = os.path.join(input_dir,"laz")
download_dir = os.path.join(input_dir,"Downloads")


#%% Input files
data_url_list_filename = os.path.join(input_dir,"data_links.txt")


#%% files
redis_url = None
local_keystore_filename = os.path.join(input_dir,"keystore.json")


#%% variables
keystore_ini_dict = {"Downloaded":[]}