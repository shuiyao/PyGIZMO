def halogas_components():
    pass

def radial_profile():
    pass

def wind_fraction():
    pass

def get_halo_metallicity():
    pass

# show_massz, if table not found, create
# galaxy metallicity: SF-weighted, star metallicity
# Get a new file gal_z.attrs {logM*, logMgal, logMgas, logMhalo, ZSF, Sfr}

import units
from astroconst import pc, ac
units_tipsy = units.Units('tipsy', lbox_in_Mpc=12.0)

def get_galaxy_mass_metallicity_relation(snap, oxygen=True, sfr_weighted=True):
    Metal = 'O' if (oxygen) else 'Zmet'
    Weight = 'Sfr' if (sfr_weighted) else 'Mass'

    snap.load_gas_particles(['galId',Weight,Metal])
    snap.load_galaxies(['Mtot','Mgas','Mstar','Sfr'])

    snap.gp['norm'] = snap.gp[Weight] * snap.gp[Metal]
    gal_attrs = snap.gp.groupby('galId').sum()
    gal_attrs['Zgal'] = gal_attrs['norm'] / gal_attrs[Weight]

    snap.gals['LogMgal'] = np.log10(snap.gals['Mtot'] * units_tipsy.m / ac.msolar)
    snap.gals['LogMgas'] = np.log10(snap.gals['Mgas'] * units_tipsy.m / ac.msolar)
    snap.gals['LogMstar'] = np.log10(snap.gals['Mstar'] * units_tipsy.m / ac.msolar)

    gal_attrs = pd.merge(gal_attrs['Zgal'],
                         snap.gals[['LogMgal','LogMgas','LogMstar','Sfr']],
                         left_index=True, right_index=True)
    return gal_attrs

def get_galaxy_cold_gas_fraction():
    pass
