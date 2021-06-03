# ETL procedure for simulation outputs (snapshots) in HDF5 format.
# Extract selected fields (density, temperatures, etc.) from HDF5 by chunks
# Transform selected fields into nested parquet data structure
# Write single parquet file for each snapshot.

import h5py
import os
from myinit import *
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pygizmo import galaxy

# model = "l25n144-phew-rcloud"
model = "l25n144-phew-rcloud"
path_model = os.path.join(DIRS['DATA'], model)
# path_hdf5 = os.path.join(path_model, "snapshot_108.hdf5")
path_schema = "schema_hdf5.csv"
# Cast PandaType from the schema file into PyArrow format
type_cast_dict = {
    'int32':pa.int32(),
    'int64':pa.int64(),
    'float64':pa.float64(),
    'float32':pa.float32()
}

def convert_hdf5_to_parquet(path_hdf5, snapnum, path_parquet=None, partitions=1, path_schema="schema_hdf5.csv"):
    '''
    Convert a single HDF5 snapshot into parquet.
    
    Parameters
    ----------
    path_hdf5 : string
        Path to the data folder containing the HDF5 files.
    path_parquet: string, default None
        The output folder for the parquet file.
        If None, set to path_hdf5.
    snapnum : int
        The index of the snapshot to convert.
    partitions: int, default 1
        The number of iterations to read the HDF5 snapshot if it is too large.
        The function will extract and transform the snapshot by chunks.
        By default, ETL the entire HDF5 file in one go.

    Returns
    -------
    None

    Example
    -------
    >>> pschema = convert_hdf5_to_parquet("/simdata/l25n144p", 108, partitions=4)
    '''

    filename = "snapshot_{:03d}.hdf5".format(snapnum)
    filename = os.path.join(path_hdf5, filename)
    if(path_parquet is None):
        path_parquet = path_hdf5
    fout = os.path.join(path_parquet, "gaspart_{:03d}.parquet".format(snapnum))
    hf = h5py.File(filename, "r")
    gp = hf['PartType0']
    ngas = hf['Header'].attrs['NumPart_Total'][0]
    schema = pd.read_csv(path_schema, header=0).set_index('FieldName')
    ibeg = list(range(0, ngas, int(ngas/partitions)))
    ibeg.append(ngas)
    print(ngas)

    # Do each partition one-by-one
    # Future work: multi-thread I/O
    for ipart in range(partitions):
        arrs, names, fields = [], [], []
        l, r = ibeg[ipart], ibeg[ipart+1]
        print("Load Partition {:3d}: [{:8d}, {:8d})".format(ipart, l, r))
        # Step 0. Write snapnum as a column
        # --------------------------------
        arrs.append(np.array([snapnum]*(r-l), dtype='int'))
        field = ('snapnum', type_cast_dict['int32'])
        names.append('snapnum')
        fields.append(field)
        
        # Step 1. Select Fields from HDF5
        # --------------------------------
        # Load selected columns one-by-one
        for idx in schema.index:
            hdf5field = schema.loc[idx].Hdf5FieldName
            dtype = schema.loc[idx].PandasType
            if(hdf5field not in gp):
                if(ipart == 0):
                    print("{} is not found in the HDF5 file.".format(hdf5field))
                continue
            if(hdf5field == "Coordinates"):
                tmp = gp[hdf5field][l:r].astype(dtype)
                arr = pa.StructArray.from_arrays(
                    (tmp[:,0], tmp[:,1], tmp[:,2]),
                    names=("x","y","z"))
                field = (idx, arr.type)
            elif(hdf5field == "Metallicity"):
                arr = gp[hdf5field][l:r,0].astype(dtype)
                field = (idx, type_cast_dict[dtype])
            else:
                arr = gp[hdf5field][l:r].astype(dtype)
                field = (idx, type_cast_dict[dtype])
            arrs.append(arr)
            names.append(idx)
            fields.append(field)

        # Step 2. Obtain the ID of the host haloes from Galaxy Data
        # --------------------------------
        sogrpname = "so_z{:03d}.sogrp".format(snapnum)
        sogrpname = os.path.join(path_hdf5, sogrpname)
        if(os.path.exists(sogrpname)):
            hids = galaxy.read_sogrp(sogrpname, gheader=hf['Header'], gas_only=True)
            hids = hids.to_numpy(dtype='int').flatten()    
            field = ('haloId', type_cast_dict['int32'])
            arrs.append(hids[l:r])
            names.append('haloId')
            fields.append(field)
        else:
            print("{} is not found. Skip the haloId field.".format(sogrpname))

        if(ipart == 0):
            pschema = pa.schema(fields)
            pwriter = pq.ParquetWriter(fout, pschema)
        # tab = pa.Table.from_arrays(tuple(arrs), names=tuple(names))
        tab = pa.Table.from_arrays(tuple(arrs), schema=pschema)
        pwriter.write_table(tab)
    pwriter.close()
    return pschema

pschema = convert_hdf5_to_parquet(path_model, 108, "output", partitions=4)
# Test
# df = spark.read.parquet("snap_108.parquet")
# pf = pq.read_table("test.parquet")
# pf, pf.column_names, pf.schema, pf.nbytes, pf.num_rows
# df = pd.read_parquet("l25n144-phew-rcloud-108.parquet")
