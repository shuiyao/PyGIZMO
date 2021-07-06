import json


class SimLog():
    '''
    Manager of the log file for a simulation. The log file keeps tracks of 
    the operations (create, update, remove) on the derived results from a 
    simulation (i.e., DerivedTables)
    '''
    
    def __init__(self, model, path_log):
        self._model = model
        self._path = path_log
        self._records = {}

    def load(self):
        with open(self._path, "r") as f:
            self._records = json.load(f)

    def dump(self):
        with open(self._path, "w") as f:
            json.dump(self._records, f, indent=2)

    def delete_event(self, name):
        self._records.pop(name, None)

    def write_event(self, name, params):
        delete_event(name)
        params['lastupdate'] = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self._records[name] = params

        
