import os
import time
from pathlib import Path

#%% configurations for data and input/output directories
base_dir = os.path.join(os.getcwd(),os.pardir)
data_dir = os.path.join(base_dir,"data")
input_dir = os.path.join(data_dir,"input")
input_laz_dir = os.path.join(input_dir,"laz")
download_dir = os.path.join(input_dir,"Downloads")
dtm_dir = os.path.join(input_dir,"dtm")
debug_csv_dir = os.path.join(input_dir,"csv")
debug_subset_dir = os.path.join(input_dir,"subsets")


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
chunk_size = 1_000_000  # chunk size for reading las file
subset_n  = 10

#%%
def log(message):
    print(message)
    with open(logfile,"a") as log_file:
        log_file.write(f"{time.ctime()}  {message} \n")

#%% make required dirs
required_dirs = [
        data_dir,
        input_dir,
        download_dir,        
        input_laz_dir,
        dtm_dir,
        debug_csv_dir,
        debug_subset_dir
        ]
    
for dir1 in required_dirs:
    Path(dir1).mkdir(exist_ok=True)