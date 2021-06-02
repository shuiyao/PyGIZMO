import numpy as np
import pandas as pd

# TODO: Build the gpTable and initTable
# gpTable can be temporary, unless we can handle a gigantic table with Spark

# Procedure:
# 1a. Identify the gas particles we want to track.
#    - Want to know gp.mwind = {'SELF':, 'SAT':, ...}
# 1b. Create gpTable (n_gp * n_snapnum)
# 2. Track each of them back in time, compile a list of halos to do. (haloId, snapnum)
# 3a. For each (halo, snapnum), get all PhEWs in it
# 3b. Assign to each PhEW, snapnum_init, haloId_init, host_status (['SELF', 'SAT', ...])
# 3c. For each (halo, snapnum), compute mgain = {'SELF':, 'SAT'}
# 4. Track gp through time again, accumulating dmass (x) (halo, snapnum).mgain

def load_timeinfo_for_snapshots():
    df_redz = pd.read_csv("redshifts.txt", sep='\s+', header=0)
    df_redz = df_redz.drop("#snapnum", axis=1)
    return df_redz

# Find the first snapshot after ainit
snapnum_first = (df_redz.a >= ainit).argmax()

# Select all PhEW particles inside a given halo at snapnum
gpTable.query("snapnum==@snapnum AND hostId==@hostId AND Mc > 0")

def gp_mass_gain_since_last_snapshot(PId, snapnum, gpTable):
    '''
    Find the amount of mass gained in a normal gas particle (PId) between the 
    previous snapshot (snapnum-1) and the current snapshot (snapnum)
    '''
    # TODO: What if the gas particle gets accreted?
    # TODO: What if the gas particle gets splitted?
    # TODO: What if the gas particle hasn't appeared in the previous snapshot?
    if(snapnum == 0): return 0.0
    mass_this = gpTable.query('snapnum==@snapnum AND PId==@PId').Mass
    mass_last = gpTable.query('snapnum==@snapnum-1 AND PId==@PId').Mass
    return mass_this - mass_last

def phew_mass_loss_since_last_checkpoint(PId, snapnum, gpTable, initTable):
    '''
    Find the amount of mass loss for a PhEW particle (PID) between the 
    last checkpoint MAX(a{snapnum-1}, ainit). 

    Parameters
    ----------
    PId: int
        The Particle ID of the PhEW particle
    snapnum: int
        The end time for query.
    gpTable: pandas.DataFrame.
        The table for all gas particles
        Columns: PId*, snapnum, Mass, haloId, ...
    initTable: pandas.DataFrame.
        The table for the initwinds information
        Columns: PId*, ainit, minit

    Return
    ------
    Mass loss of the PhEW particle during the given period of time.
    '''

    mass_this = gpTable.query('snapnum==@snapnum AND PId==@PId').Mass
    ainit = float(initTable.query('PId==@PId').ainit)
    minit = float(initTable.query('PId==@PId').minit)
    if(redzTable.loc[snapnum-1].a < ainit):
        mass_last = minit
    else:
        mass_last = gpTable.query('snapnum==@snapnum-1 AND PId==@PId').Mass
    return mass_this - mass_last

# Mass Change From Last Checkpoint
# 1. Normal Gas Particle: dMass = Mwind[i] - Mwind[i-1] or Mwind[acc] - Mwind[i-1]
#    (snapnum, PID) -> Mgain
# 2. PhEW Particle: dMass = Mass[i] - Mass[i-1] or Mass[i] - Mass[init]
#    (snapnum, PID) -> Mloss
