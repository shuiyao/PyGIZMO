'''
Connectors to specexbin, the mock spectra generation engine.
Specexbin generates mock absorption spectra in two modes:
I. Long spectra: 
   Covers a long redshift/velocity space that needs stacking of 
   multiple snapshots.
II. Short spectra:
   Usually for sightlines that are close to a galaxy with the 
   purpose of probing the gas structures near the galaxy. Covers 
   a short distance. Use a single snapshot but need to specify 
   the coordinates of the sightline.
'''
