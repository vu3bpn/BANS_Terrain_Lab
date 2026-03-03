import pandas as pd
import json
import os
from pathlib import Path
from config import *

class keystore:
    def __init__(self):
        if os.path.exists(local_keystore_filename):
            self.store = json.loads(Path(local_keystore_filename).read_text())
        else:
            self.store = keystore_ini_dict
    def set(self, key, value):
        self.store[key] = value
        
    def add_to_list(self,key,value):
        self.store[key].append(value)
        
    def get(self, key):
        if not key in self.store:
            self.store[key] = []
        return self.store[key]
    
    def flush(self):
        with open(local_keystore_filename,'w') as key_file:
            key_file.write(json.dumps(self.store,indent=2))
    
store = keystore()