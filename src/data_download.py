import os 
from pathlib import Path
import zipfile
from config import *
from keystore import *

def download_data():
    Path(download_dir).mkdir(exist_ok=True)
    list_of_urls = open(data_url_list_filename).read().splitlines()
    for link1 in list_of_urls:
        if link1 not in store.get("Downloaded"):
            log(f"downloading {link1}")
            os.system(f"wget --no-check-certificate  -c -q -P {download_dir} {link1}")
            store.add_to_list("Downloaded",link1)
            store.flush()
            
def extract_data():
    dl_file_list = os.listdir(download_dir)
    for file1 in dl_file_list:
        if file1 in store.get("Extracted"):
            continue
        with zipfile.ZipFile(os.path.join(download_dir,file1),'r') as zip_file:
            zip_file.extractall(input_laz_dir)
        store.add_to_list("Extracted",file1)
    store.flush()
            
if __name__ == "__main__":
    download_data()
    extract_data()