'''
General utilities.
'''

import json
import os
import sys
from config import cfg

path_schema = os.path.join(cfg['Paths']['pygizmo'], "schema.json")

def cumhist(arr, bins=10, weights=None, reverse=False):
    '''
    Calculate the cumulative distribution of a 1-D array of values.

    Parameters
    ----------
    arr : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
    weights : array_like, optional
    reverse : bool, default: False
        If True, returns the array in reverse order.

    Returns
    -------
    hist : array
        The cumulated value at each edge point.
    bin_edges : array of dtype float
        The bin edges
    '''
    from numpy import histogram
    hist, edges = histogram(arr, bins=bins, weights=weights)
    cen = 0.5*(edges[1:] + edges[:-1])
    cum = [0]
    for value in hist:
        cum.append(cum[-1]+value)
    return array(cum), edges
    
def load_default_schema():
    '''
    Load the default schemas for a couple of simulation outputs.
    The default schemas are saved in a JSON file named "schema.json"
    '''
    with open(path_schema) as json_data:
        schema = json.load(json_data)
    return schema

def talk(text, verbose_level='always', err=False):
    if(isinstance(verbose_level, str)):
        verbose_level = cfg['Verbose'][verbose_level.lower()]
    if(verbose_level > cfg['Verbose']['default']):
        return
    dest = sys.stderr if err else sys.stdout
    print(text, file=dest)
