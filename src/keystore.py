import pandas as pd
import json
from config import *

class keystore:
    def __init__(self):
        if os.path.exists(local_keystore_filename):
            self.store = json.loads(open(local_keystor_filename,'r').read())
        self.store = keystore_ini_dict
    def set(self, key, value):
        self.store[key] = value
        
    def add_to_list(self,key,value):
        self.store[key].append(value)
        
    def get(self, key):
        if not key in self.store:
            self.store[key] = []
        return self.store[key]
    def __del__(self):
        open(local_keystor_filename,'w').write(json.dumps(self.store))
store = keystore()