__all__ = ["compute_halo_gas_components", "radial_profile", "wind_fraction", "get_halo_metallicity", "derive_galaxy_properties"]

from astroconst import pc, ac
from config import SimConfig
import pandas as pd
from simulation import Simulation
from snapshot import Snapshot

import abc

class SimAnalysis(abc.ABC):
    def __init__(self, model, config=SimConfig()):
        self._model = model
        self._cfg = config
        self._sim = Simulation(model)

    def compute(self, z, overwrite=False):
        '''
        Compute the DerivedTable(s) z(s).

        Parameters
        ----------
        z: float or list.
            The redshift at which to generate the GSMF.
        '''
        if(isinstance(z, float)):
            z = [z]
        for redz in z:
            tab = self._get_table_at_z(redz)
            tab.build_table(overwrite=overwrite)
            
    def load(self, z):
        '''
        Load the DerivedTable at a redshift z.
        '''
        tab = self._get_table_at_z(z)
        return tab.load_table()

    @abc.abstractmethod
    def _get_table_at_z(z):
        pass

    @property
    def model(self):
        return self._model


class Gsmf(SimAnalysis):
    def __init__(self, model):
        super(Gsmf, self).__init__(model)

    def _get_table_at_z(z):
        snapnum = self._sim.find_snapnum(z, 'closest')
        snap = Snapshot(self._model, snapnum)
        return GsmfTable(snap)

class Smhm(SimAnalysis):
    def __init__(self, model):
        super(Smhm, self).__init__()

    def _get_table_at_z(z):
        snapnum = self._sim.find_snapnum(z, 'closest')
        snap = Snapshot(self._model, snapnum)
        return SmhmTable(snap)

class Mzr(SimAnalysis):
    def __init__(self, model):
        super(Mzr, self).__init__()

    def _get_table_at_z(z):
        snapnum = self._sim.find_snapnum(z, 'closest')
        snap = Snapshot(self._model, snapnum)
        return GalaxyAttributes(snap)

    def load(z):
        tab = self._get_table_at_z(z)
        df = tab.load_table()
        return df[['logMstar', 'Zgal']]



def compute_halo_gas_components(snap, Tcut=None):
    '''
    Decompose the baryons in each halo into several phases, including cold gas,
    hot gas, star-forming interstellar gas and stars. 

    Parameters
    ----------
    snap: Snapshot object.
    Tcut: float. Default=None
        The log temperature threshold that separates cold gas from hot gas.
        If None. Use the value from the config file (typically 5.5).

    Returns
    -------
    halos: pandas.DataFrame
        The mass within each baryon phase for all the halos.
        columns: logMsub, logMvir, Mcold, Mhot, Mism, Mstar
    '''

    cfg = SimConfig()
    if(Tcut == None):
        Tcut = float(cfg.get('Default','logT_threshold'))

    snap.load_gas_particles(['haloId','Mass','Sfr','logT'])
    snap.load_star_particles(['haloId','Mass'])
    snap.load_halos(['Mvir','Msub'])

    grps = snap.gp.groupby('haloId')
    # Separate baryonic particles into cold, hot, ISM and star
    mism = grps.apply(lambda x : (x.Mass * (x.Sfr > 0)).sum()).rename('Mism', inplace=True)
    mcold = grps.apply(lambda x : (x.Mass * ((x.Sfr == 0) & (x.logT < Tcut))).sum()).rename('Mcold', inplace=True)
    mhot = grps.apply(lambda x : (x.Mass * ((x.Sfr == 0) & (x.logT > Tcut))).sum()).rename('Mhot', inplace=True)
    mstar = snap.sp.groupby('haloId').sum()['Mass'].rename('Mstar', inplace=True)
    m_comp = pd.concat([mcold, mhot, mism, mstar], axis=1)
    halos = pd.merge(snap.halos, m_comp, how='left', left_index=True, right_index=True)
    halos.fillna(0.0, inplace=True)
    return halos

def radial_profile():
    ''' 
    Compute radial profiles of a galactic halo.
    '''
    pass

def wind_fraction():
    pass

def get_halo_metallicity():
    pass

def derive_galaxy_properties(snap, oxygen=True, sfr_weighted=True, save=False):
    '''
    Derive several properties, including the star formation rate, weighted 
    total metallicity, gas mass, stellar mass and total mass of all galaxies 
    in a snapshot. Store the result in a Pandas DataFrame.

    Parameters
    ----------
    snap: The Snapshot object.
    oxygen: boolean. Default = True.
        If True, represent galaxy metallicity with the oxygen abundance.
        If False, use the total metallicity.
    sfr_weighted: boolean. Default = True.
        If True, calculate the galaxy metallicity weighted by star formation 
        rate. Otherwise, calculate the galaxy metallicity weighted by mass.
    save: boolean. Default = False.
        If True, save the results into a CSV file in the workdir.

    Return
    ------
    gal_attrs: pandas.DataFrame.
        columns: Zgal, LogMgal, LogMgas, LogMstar, Sfr.
        Several plotting scripts will use this output.
    '''
    element = 'O' if (oxygen) else 'Zmet'
    weight = 'Sfr' if (sfr_weighted) else 'Mass'

    snap.load_gas_particles(['galId',weight,element])
    snap.load_galaxies(['Mgal','Mgas','Mstar','Sfr'])

    snap.gp['norm'] = snap.gp[weight] * snap.gp[element]
    gal_attrs = snap.gp.groupby('galId').sum()
    gal_attrs['Zgal'] = gal_attrs['norm'] / gal_attrs[weight]

    gal_attrs = pd.merge(gal_attrs['Zgal'],
                         snap.gals[['logMgal','logMgas','logMstar','Sfr']],
                         left_index=True, right_index=True)

    if(save):
        gal_attrs.to_csv(os.path.join(snap._path_workdir, "gals_z{}.attrs"))
    return gal_attrs

