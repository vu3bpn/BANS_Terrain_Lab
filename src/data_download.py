import os 
from pathlib import Path
from config import *
from keystore import *

def download_data():
    Path(download_dir).mkdir(exist_ok=True)
    list_of_urls = open(data_url_list_filename).read().splitlines()
    for link1 in list_of_urls:
        if link1 not in store.get("Downloaded"):
            os.system(f"wget --no-check-certificate  -P {download_dir} {link1}")
            store.add_to_list("Downloaded",link1)
            
if __name__ == "__main__":
    download_data()