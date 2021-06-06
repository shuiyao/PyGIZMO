'''
General utilities.
'''
__all__ = ['path_schema', 'cumhist', 'talk', 'load_timeinfo_for_snapshots']

import json
import os
import sys
from config import cfg
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

path_schema = os.path.join(cfg['Paths']['pygizmo'], cfg['Schema']['galaxy'])

pyArrowTypeCast = {
    "int64":pa.int64(),
    "int32":pa.int32(),
    "float32":pa.float32(),
    "float64":pa.float64()
}

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

def read_parquet_schema(source):
    """
    Return the schema of a parquet file.

    Returns
    -------
    schemaParquet: pandas.DataFrame
        Columns: column, pa_dtype
    """
    # Ref: https://stackoverflow.com/a/64288036/
    if(os.path.isdir(source)):
        talk("Parquet source is a directory, search within the folder.", "quiet")
        source = glob.glob(source+"/*.parquet")[0]
    schema = pq.read_schema(source, memory_map=True)
    schema = pd.DataFrame(({"column": str(name), "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
    schema = schema.reindex(columns=["column", "pa_dtype"], fill_value=pd.NA)
    # Ensures columns in case the parquet file has an empty dataframe.
    return schema

def pyarrow_read_schema(schema_json):
    '''
    Create a pyArrow schema from JSON type schema definition.

    Parameters
    ----------
    schema_json: dict.
        Must have a 'columns' field that gives the order of fields
        Must have a 'dtypes' field that maps fieldname to numpy dtype
        Example: {'columns':['col1', 'col2'], 
                  'dtypes':{'col1':int32, 'col2':int64}}

    Returns
    -------
    schema: pyArrow Schema.
    '''
    
    cols = schema_json['columns']
    dtypes = schema_json['dtypes']
    fields = []
    for col in cols:
        field = pa.field(col, pyArrowTypeCast[dtypes[col]])
        fields.append(field)
    return pa.schema(fields)

def load_timeinfo_for_snapshots(fredz="redshift.txt"):
    df_redz = pd.read_csv(fredz, sep='\s+', header=0)
    df_redz = df_redz.drop("snapnum", axis=1)
    return df_redz
