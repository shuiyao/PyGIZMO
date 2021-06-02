'''
Procedures related to finding the progenitors of galactic halos in previous 
snapshots.
'''

import snapshot
from utils import talk
import galaxy
from myinit import *
import pandas as pd

__all__ = ['find_all_previous_progenitors', 'find_progenitors',
           'build_haloId_hostId_map', 'get_relationship_between_halos']

def find_all_previous_progenitors(snap, overwrite=False):
    '''
    Find the progenitors for all halos within a snapshot in all previous 
    snapshots.

    Parameters
    ----------
    snap: class Snapshot.
    overwrite: boolean. Default=False.
        If False, first try to see if a table already exists. Create a new 
        table if not.
        If True, create a new table and overwrite the old one if needed.
    
    Returns
    -------
    progTab: pandas DataFrame.
        A pandas table storing information for the progenitors of halos at
        different time.
        columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
    
    Examples
    --------
    >>> snap = snapshot.Snapshot('l12n144-phew', 100)
    >>> progTab = find_all_previous_progenitors(snap, overwrite=True)

    '''

    schema = {'columns':['haloId','snapnum','progId','hostId','logMvir','logMsub'],
              'dtypes':{'haloId':'int32',
                        'snapnum':'int32',
                        'progId':'int32',
                        'hostId':'int32',                        
                        'logMvir':'float32',
                        'logMsub':'float32'
              }
    }
    fout = os.path.join(snap._path_workdir, "progens_{:03d}.csv".format(snap.snapnum))
    if(os.path.exists(fout) and overwrite == False):
        talk("Read existing progenitor file: {}".format(fout), 'normal')
        return pd.read_csv(fout, skiprows=1, names=schema['columns'], dtype=schema['dtypes'])

    progTab = None
    for snapnum in range(snap.snapnum-3, snap.snapnum):
    # for snapnum in range(0, snap.snapnum):        
        talk("Finding progenitors in snapshot {:03d}".format(snapnum), 'normal')
        snapcur = snapshot.Snapshot(snap.model, snapnum)
        snapcur.load_halos(['Mvir', 'Msub'])
        haloId2hostId = galaxy.read_sopar(snap._path_sopar, as_dict=True)
        haloId2progId = find_progenitors(snap, snapcur)
        df = pd.DataFrame(index=haloId2progId.keys())
        df['snapnum'] = snapnum
        df['progId'] = df.index.map(haloId2progId)
        df['hostId'] = df.progId.map(haloId2hostId)
        df = pd.merge(df, snapcur.halos, how='left', left_on='progId', right_index=True)
        progTab = df.copy() if (progTab is None) else pd.concat([progTab, df])
    progTab.index.rename('haloId', inplace=True)
    progTab.reset_index().to_csv(fout, index=False, columns=schema['columns'])
    return progTab

def find_progenitors(snap, snap_early):
    '''
    For each halo from a snapshot, find its main progenitor in some early 
    snapshot. The main progenitor of a halo is defined as the halo that hosts 
    a majority of its current dark matter particles.

    Algorithm:
    1. Select dark particles that was found in a halo in both the early and the current snapshot (hostId > 0)
    2. Find the total number of dark particles, in any hostId that was from progId
    3. Pick the progId with max(count) as the progenitor of hostId

    Parameters
    ----------
    snap: class Snapshot.
    snap_early: class Snapshot.
        A snapshot at some earlier time in which the progenitors are found.

    Returns
    -------
    haloId2progId: dict.
        A temporary mapping from snap.haloId to snap_early.progId.
        Note: not all haloId will find a progenitor.
    '''

    snap.load_dark_particles()
    snap_early.load_dark_particles()
    dp = snap.dp[snap.dp.haloId > 0].set_index('PId')
    dpe = snap_early.dp[snap_early.dp.haloId > 0]\
                    .set_index('PId')\
                    .rename(columns={'haloId':'progId'})
    dp = pd.merge(dp, dpe, how='inner', left_index=True, right_index=True)
    grp = dp.reset_index().groupby(['haloId','progId']).count().reset_index()
    idx = grp.groupby('haloId')['PId'].transform(max) == grp['PId']
    return dict(zip(grp[idx].haloId, grp[idx].progId))

def build_haloId_hostId_map(snap):
    '''
    Build maps between haloId and hostId for each snapshot before the snapshot
    parsed in the arguments.

    Parameter
    ---------
    snap: class Snapshot.

    Returns
    -------
    hostMap: dict.
        {snapnum:{haloId:hostId}}
    '''
    hostMap = dict()
    for snapnum in range(0, snap.snapnum):
        snapcur = snapshot.Snapshot(snap.model, snapnum)
        haloId2hostId = galaxy.read_sopar(snap._path_sopar, as_dict=True)
        hostMap[snapnum] = haloId2hostId
    return hostMap

def get_relationship_between_halos(haloId, haloIdTarget, snapnum, progTab, hostMap):
    '''
    Get the relationship between the progenitor of a halo and another halo in 
    the same snapshot as the progenitor.

    Parameters
    ----------
    haloId: int.
        The haloId of the halo in the current snapshot.
    haloIdTarget: int
        The haloId of the target halo at an earlier time.
    snapnum: int
        The corresponding snapnum of the snapshot target halo.
    progTab: pandas DataFrame.
        Output of find_all_previous_progenitors().
        Defines the progenitors of a halo in any previous snapshot.
        Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
    hostMap: dict.
        Output of build_haloId_hostId_map()
        Mapping between haloId and hostId for each snapshot.

    Returns
    -------
    relation: str
        One of ['SELF', 'PARENT', 'SAT', 'SIB', 'IGM']
        Interpreted as: haloIdTarget at snapnum is the [relation] of the 
        progenitor of haloId at that snapshot.

    Example
    -------
    >>> snap = snapshot.Snapshot('l12n144-phew', 100)
    >>> hostMap = build_haloId_hostId_map(snap)
    >>> progTab = find_all_previous_progenitors(snap)
    >>> get_relationship_between_halos(10, 30, 50, progTab, hostMap)
    '''

    prog = progTab.loc[haloId].query('snapnum==@snapnum')
    if(prog.empty): return "IGM"
    if(int(prog.progId) == haloIdTarget): return "SELF"
    if(int(prog.hostId) == haloIdTarget): return "PARENT"
    hostIdTarget = hostMap[snapnum][haloIdTarget]
    if(int(prog.progId) == hostIdTarget): return "SAT"
    if(int(prog.hostId) == hostIdTarget): return "SIB"
    return "IGM"

