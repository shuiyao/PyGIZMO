'''
Procedures related to finding the progenitors of galactic halos in previous 
snapshots.
'''

import snapshot
from utils import talk
import galaxy
import pandas as pd
from tqdm import tqdm
import os

schema_progtable = {'columns':['haloId','snapnum','progId','hostId','logMvir','logMsub'],
                    'dtypes':{'haloId':'int32',
                              'snapnum':'int32',
                              'progId':'int32',
                              'hostId':'int32',
                              'logMvir':'float32',
                              'logMsub':'float32',
                              'Npart':'int32'}
}

class ProgTracker():

    @staticmethod
    def build_haloId_hostId_map(simulation, overwrite=False):
        '''
        Build maps between haloId and hostId for each snapshot before the snapshot
        parsed in the arguments.

        Parameter
        ---------
        simulation: class Simulation

        Returns
        -------
        hostmap: pandas.DataFrame
            Columns: snapnum*, haloId*, hostId
        '''
        fout = os.path.join(simulation._path_workdir, "hostmap.csv")
        if(os.path.exists(fout) and overwrite == False):
            talk("Read existing hostmap file: {}".format(fout), 'normal')
            hostmap = pd.read_csv(fout, header=0)
            return hostmap.set_index(['snapnum', 'haloId'])

        frames = []
        for snapnum in range(0, simulation.nsnaps):
            snapcur = snapshot.Snapshot(simulation.model, snapnum)
            haloId2hostId = galaxy.read_sopar(snapcur._path_sopar, as_dict=True)
            # haloId2hostId may be empty when no galaxy existed
            df = pd.DataFrame({'snapnum':snapnum,
                               'haloId':haloId2hostId.keys(),
                               'hostId':haloId2hostId.values()})
            frames.append(df)
        hostmap = pd.concat(frames, axis=0).set_index(['snapnum', 'haloId'])
        hostmap.to_csv(fout)
        return hostmap

    @staticmethod
    def get_relationship_between_halos(haloIdTarget, haloId, snapnum, progtable, hostmap):
        '''
        Get the relationship between the progenitor of any halo from a snapshot and 
        another halo in the same snapshot as the progenitor. 

        Parameters
        ----------
        haloIdTarget: int.
            The haloId of the halo in the current snapshot.
        haloId: int
            The haloId of the target halo at an earlier time.
        snapnum: int
            The corresponding snapnum of the snapshot target halo.
        progtable: pandas DataFrame.
            Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
            Output of find_all_previous_progenitors().
            Defines the progenitors of any halo in any previous snapshot.
        hostmap: dict.
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
        >>> hostmap = build_haloId_hostId_map(snap)
        >>> progtable = find_all_previous_progenitors(snap)
        >>> get_relationship_between_halos(10, 30, 50, progtable, hostmap)
        '''

        prog = progtable.loc[haloIdTarget].query('snapnum==@snapnum')
        if(prog.empty): return "IGM"
        if(int(prog.progId) == haloId): return "SELF"
        if(int(prog.hostId) == haloId): return "PARENT"
        hostIdTarget = hostmap[snapnum][haloId]
        if(int(prog.progId) == hostId): return "SAT"
        if(int(prog.hostId) == hostId): return "SIB"
        return "IGM"

    @staticmethod
    def compile_halos_hosts(data, fields=['snapnum', 'haloId']):
        '''
        Find all unique halos (snapnum, haloId) from a table.

        Parameters
        ----------
        data: pandas.DataFrame.
            A table, each entry of which maps to a halo.
            It could be from a gptable or pptable, where each particle has been
            assigned to a unique halo.
        fields: []
            The name of the fields that correspond to the snapnum and haloId in 
            the input table.
        '''
        
        halos = data.loc[:, fields]
        halos.rename(columns={fields[0]:'snapnum', fields[1]:'haloId'}, inplace=True)
        halos = halos[halos.haloId != 0].drop_duplicates()
        return halos

    @staticmethod
    def assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap):
        '''
        Map from the unique halo identifier (snapnum, haloId) to a descriptor for 
        its relationship with another halo (haloIdTarget) at a later time.
        In the accretion tracking engine, the table is built iteratively for each 
        halo of interest.

        Parameters
        ----------
        haloIdTarget: int.
            The haloId of the halo in the current snapshot.
        halos: pandas.DataFrame
            Columns: haloId*, snapnum*
            Halos uniquely identified with the (haloId, snapnum) pair
        progtable: pandas.DataFrame.
            Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
            Output of progen.find_all_previous_progenitors().
            Defines the progenitors of any halo in any previous snapshot.
        hostmap: pandas.DataFrame.
            Columns: snapnum*, haloId*, hostId
            Output of progen.build_haloId_hostId_map()
            Mapping between haloId and hostId for each snapshot.

        Returns:
        relation: pandas.DataFrame
            Columns: snapnum*, haloId*, relation
            Map between a halo (MultiIndex(snapnum, haloId)) to the relation, which 
            defines its relation to another halo (haloIdTarget) at a later time.
        '''

        from progen import ProgTracker

        progsTarget = progtable.loc[progtable.index == self.haloId,
                                    ['snapnum', 'progId', 'hostId']]
        progsTarget.rename(columns={'hostId':'progHost'}, inplace=True)

        halos = halos[['snapnum', 'haloId']].set_index(['snapnum', 'haloId'])

        # Find the hostId of each halo in its snapshot
        halos = halos.join(hostmap, how='left') # (snapnum, haloId) -> hostId
        halos = halos.reset_index()

        halos = pd.merge(halos, progsTarget, how='left',
                         left_on = 'snapnum', right_on = 'snapnum')

        # Call ProgTracker
        halos['relation'] = halos.apply(
            lambda x : ProgTracker.define_halo_relationship(
            x.progId, x.progHost, x.haloId, x.hostId), axis=1
        )

        return halos[['snapnum','haloId','relation']].set_index(['snapnum', 'haloId'])

    @staticmethod
    def define_halo_relationship(progId, progHost, haloId, hostId):
        if(progId == 0): return "IGM"
        if(progId == haloId): return "SELF"
        if(progHost == haloId): return "PARENT"
        if(progId == hostId): return "SAT"
        if(progHost == hostId): return "SIB"
        return "IGM"

