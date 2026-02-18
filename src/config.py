import os
import time
from pathlib import Path

#%% configurations for data and input/output directories
data_dir = os.path.join(Path.cwd().parent,"data")
input_dir = os.path.join(data_dir,"input")
input_laz_dir = os.path.join(input_dir,"laz")
download_dir = os.path.join(input_dir,"Downloads")


#%% Input files
data_url_list_filename = os.path.join(input_dir,"data_links.txt")
logfile = os.path.join(data_dir,"run_log.txt")


#%% files
redis_url = None
local_keystore_filename = os.path.join(input_dir,"keystore.json")


#%% variables
keystore_ini_dict = {"Downloaded":[]}


#%%
def log(message):
    print(message)
    open(logfile,"a").write(f"{time.ctime()}  {message}")

#%% make required dirs
required_dirs = [
        download_dir,
        input_dir,
        input_laz_dir,
        ]
    
for dir1 in required_dirs:
    Path(dir1).mkdir(exist_ok=True)