from astroconst import pc, ac
from config import cfg
import pandas as pd

__mode__ = "__X__"
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
    if(Tcut == None):
        Tcut = float(cfg['Default']['logT_threshold'])

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

if __mode__ == "__test__":
    from snapshot import snap
    h = compute_halo_gas_components(snap)
