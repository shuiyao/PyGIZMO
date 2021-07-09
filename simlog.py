import json
from config import SimConfig
from datetime import datetime
import os

class SimLog():
    '''
    Manager of the log file for a simulation. The log file keeps tracks of 
    the operations (create, update, remove) on the derived results from a 
    simulation (i.e., DerivedTables)
    '''
    
    def __init__(self, model, path_log=None, config=SimConfig()):
        self._model = model
        self._cfg = config
        if(path_log is None):
            path_log = os.path.join(self._cfg.get('Paths','workdir'),
                                    model,
                                    "derivedtables.log")
        self._path = path_log
        self._records = {}
        self.load_or_create()

    def load_or_create(self):
        if(os.path.exists(self._path)):
            with open(self._path, "r") as f:
                self._records = json.load(f)
        else:
            self.dump()

    def dump(self):
        with open(self._path, "w") as f:
            json.dump(self._records, f, indent=2)

    def load_event(self, name):
        return self._records.get(name)

    def delete_event(self, name):
        self._records.pop(name, None)

    def write_event(self, name, params):
        self.delete_event(name)
        params['lastupdate'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self._records[name] = params
        self.dump()
        
