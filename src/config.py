import os
import time
from pathlib import Path

#%% configurations for data and input/output directories
base_dir = r"/mnt/nvme1n1/Bipin/Scripts/geo_ai"
data_dir = os.path.join(base_dir,"data")
input_dir = os.path.join(data_dir,"input")
input_laz_dir = os.path.join(input_dir,"laz")
download_dir = os.path.join(input_dir,"Downloads")
dtm_dir = os.path.join(input_dir,"dtm")


#%% Input files
data_url_list_filename = os.path.join(input_dir,"data_links.txt")
logfile = os.path.join(data_dir,"run_log.txt")


#%% files
redis_url = None
local_keystore_filename = os.path.join(input_dir,"keystore.json")


#%% variables
keystore_ini_dict = {"Downloaded":[]}
dtm_resolution = 0.1
pipeline_cuncurrent_jobs = 3


#%%
def log(message):
    print(message)
    open(logfile,"a").write(f"{time.ctime()}  {message} \n")

#%% make required dirs
required_dirs = [
        download_dir,
        input_dir,
        input_laz_dir,
        dtm_dir,
        ]
    
for dir1 in required_dirs:
    Path(dir1).mkdir(exist_ok=True)