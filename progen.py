'''
Procedures related to finding the progenitors of galactic halos in previous 
snapshots.
'''

import snapshot
from myinit import *

model = "l12n144-phew-movie-200"
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
    progens: pandas DataFrame.
        A pandas table storing information for the progenitors of halos at
        different time.
        columns: haloId*, snapnum, progenId, logMvir, logMsub
    '''

    schema = {'columns':['haloId','snapnum','progenId','logMvir','logMsub'],
              'dtypes':{'haloId':'int32',
                        'snapnum':'int32',
                        'progenId':'int32',
                        'logMvir':'float32',
                        'logMsub':'float32'
              }
    }
    fout = os.path.join(snap._path_workdir, "progens_{:03d}.csv".format(snap.snapnum))
    if(os.path.exists(fout) and overwrite == False):
        print("Read existing progenitor file: {}".format(fout))
        return pd.read_csv(fout, skiprows=1, names=schema['columns'], dtype=schema['dtypes'])

    progen = None
    for snapnum in range(snap.snapnum-3, snap.snapnum):
    # for snapnum in range(0, snap.snapnum):        
        print("Finding progenitors in snapshot {:03d}".format(snapnum))
        snapcur = snapshot.Snapshot(snap.model, snapnum)
        snapcur.load_halos(['Mvir', 'Msub'])
        haloId2progenId = find_progenitors(snap, snapcur)
        df = pd.DataFrame(index=haloId2progenId.keys())
        df['snapnum'] = snapnum
        df['progenId'] = df.index.map(haloId2progenId)
        df = pd.merge(df, snapcur.halos, how='left', left_on='progenId', right_index=True)
        progen = df.copy() if (progen is None) else pd.concat([progen, df])
    progen.index.rename('haloId', inplace=True)
    progen.reset_index().to_csv(fout, index=False, columns=schema['columns'])
    return progen

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


# snap1 = snapshot.Snapshot(model, 5)
# snap2 = snapshot.Snapshot(model, 20)
# m = find_progenitors(snap2, snap1)

snap = snapshot.Snapshot(model, 20)
progen = find_all_previous_progenitors(snap, overwrite=True)
