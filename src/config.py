import os
import time
from pathlib import Path

#%% configurations for data and input/output directories
base_dir = os.path.join(os.getcwd(),os.pardir)
base_dir = os.path.normpath(base_dir)
data_dir = os.path.join(base_dir,"data")
input_dir = os.path.join(data_dir,"input")
debug_dir = os.path.join(data_dir,"debug")
input_laz_dir = os.path.join(input_dir,"laz")
download_dir = os.path.join(input_dir,"Downloads")
dtm_dir = os.path.join(input_dir,"dtm")
debug_csv_dir = os.path.join(input_dir,"csv")
debug_subset_dir = os.path.join(input_dir,"subsets")
fixed_window_subset_dir = os.path.join(input_dir,"subset_fixed_w")
split_files_subset_dir = os.path.join(input_dir,"split_files")
copc_dir = os.path.join(input_dir,"copc")
output_dir = os.path.join(data_dir,"output")
vector_dir = os.path.join(input_dir,"vectors")
model_dir = os.path.join(output_dir,"models")


#%% Input files
data_url_list_filename = os.path.join(input_dir,"data_links.txt")
logfile = os.path.join(data_dir,"run_log.txt")
data_info_file = os.path.join(output_dir,"data_info.csv")


#%% files
redis_url = None
local_keystore_filename = os.path.join(input_dir,"keystore.json")


#%% variables
keystore_ini_dict = {"Downloaded":[]}
dtm_resolution = 0.1
pipeline_cuncurrent_jobs = 2
chunk_size = 1_000_000  # chunk size for reading las file
subset_n  = 10
DTM_selected_cols = ["X","Y","Z","red","green","blue","intensity"]


las_vect_dict = {"64334_2H_(REFLIGHT)_POINT_CLOUD.las":"rajastan_64334_2h.shp"}
dtm_model_name = "DTM_transformer_model.mdl"
dtm_model_path = os.path.join(model_dir,dtm_model_name)


#%%
def log(message):
    print(message)
    with open(logfile,"a") as log_file:
        log_file.write(f"{time.ctime()}  {message} \n")

#%% make required dirs
required_dirs = [
        data_dir,
        input_dir,
        debug_dir,
        download_dir,        
        input_laz_dir,
        dtm_dir,
        debug_csv_dir,
        debug_subset_dir,
        copc_dir,
        fixed_window_subset_dir,
        split_files_subset_dir,
        output_dir,
        model_dir
        ]
    
for dir1 in required_dirs:
    Path(dir1).mkdir(exist_ok=True)