import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

from .. import snapshot

class Halo3D(object):
    '''
    Display gas particles in a selected halo.
    
    The left hand panel shows an overall view of the halo. Normal gas 
    particles have colors ranging from blue to red (plasma color map) 
    according to their log temperatures. The PhEW particles are colored in
    green, their color tunes and sizes accord with their remaining mass. The
    dense gas particles in a galaxy are displayed in light blue color.

    The right panels consist of zoomed-in views (head on and an edge on) of 
    the central region of the halo.

    Parameters
    ----------
    snap: Snapshot object.
    haloid: int. 
        The haloId of the halo to display.
    angles_faceon: [float, float]. Default = [90., 90.]
        The viewing angle for the face on view. 
    angles_edgeon: [float, float]. Default = [0., 0.]
        The viewing angle for the edge on view. 
        TODO: Can automatically figure out these angles using cross product
    rlim: float. Default = 0.8
        The boundary of the overview plot is rlim * rad, where rad is the 
        virial radius of the halo.
    rlim_zoomin: float. Default = 0.15
        The boundary of the zoomed-in views.
    cmap_logt: string. Default = "plasma"
        Color map for the normal gas particles.
    cmap_phew: string. Default = "Greens"
        Color map for the PhEW particles.
    background: string. Default = "black"
        The background color.
    figsize: (int, int). Default = (9, 6)

    Examples
    --------
    >>> snap = snapshot.Snapshot("l25n144-test", 108)
    >>> h3d = Halo3D(snap, haloid=1)

    HaloId: 1
    --------------------------------
    Center  : [   49.6,  3938.8,  5731.8]
    logMvir : 12.59
    Rvir    : 285.5 kpc

    >>> h3d.draw()
    >>> h3d.select_galaxies_by_mass_percentiles(0.96, 0.962)

           Npart    logMgal   logMstar
    galId                             
    147      478  10.873955  10.721762
    528      598  10.889726  10.810463
    594      607  10.876665  10.865065

    >>> h3d.load_halo_particles(147)

    HaloId: 147
    --------------------------------
    Center  : [11777.2,  5328.7,  5637.4]
    logMvir : 12.21
    Rvir    : 214.2 kpc

    >>> h3d.draw()

    '''
    def __init__(self, snap, haloid=None,
                 angles_faceon=[90,90], angles_edgeon=[0,0],
                 rlim=0.8, rlim_zoomin=0.15,
                 cmap_logt="plasma", cmap_phew="Greens", background="black",
                 figsize=(9,6)):
        self._snap = snap
        self._snapnum = snap.snapnum
        self.rlim = rlim
        self.rlim_zoomin = rlim_zoomin
        self.angles_faceon = angles_faceon
        self.angles_edgeon = angles_edgeon
        self.haloId = haloid

        self.path_output = snap._path_tmpdir
        self.path_figure = snap._path_figure

        if(haloid is not None):
            self.load_halo_info(haloid)
            self.load_halo_particles()

        self.cmap_logt=plt.get_cmap(cmap_logt) # 4.0 - 7.0
        self.cmap_phew=plt.get_cmap(cmap_phew) # 0.0 - 1.0
        self.cmap_star = (65./255., 105./255., 225./255., 1.0) # royal blue
        self.figsize = figsize
        self.background = background

    def load_halo_info(self, haloId):
        '''
        Load some halo information from the galaxy/halo outputs.

        Parameters
        ----------
        haloId: int.
        '''
        
        fields = ['Rvir', 'logMvir', 'x', 'y', 'z']
        if(self._snap.halos is None or
           any(item not in snap.halos.columns for item in fields)):
            self._snap.load_halos(fields=['Rvir','Mvir','x','y','z'])
        try:
            halo = self.halos.loc[haloId]
        except:
            raise KeyError("haloId {} not found in {}".format(haloId, self._snap))
        self.haloId = haloId
        self.x = self._snap._transform_coordinates(halo.x)
        self.y = self._snap._transform_coordinates(halo.y)
        self.z = self._snap._transform_coordinates(halo.z)
        self.rad = halo.Rvir
        self.mvir = halo.logMvir
        self.xlims = (self.x - self.rlim * self.rad,
                      self.x + self.rlim * self.rad)
        self.ylims = (self.y - self.rlim * self.rad,
                      self.y + self.rlim * self.rad)
        self.zlims = (self.z - self.rlim * self.rad,
                      self.z + self.rlim * self.rad)
        self.halo()

    def load_halo_particles(self, haloId=None, box=True, rewrite=False):
        '''
        Load gas particles from or near the given halo. If rewrite is not True, 
        read existing file that was created in the tmpdir.
        
        Parameters
        ----------
        haloId: int. Default = None.
            If None. Use self.haloId if exists.
            Otherwise, reset self.haloId to haloId and load the new halo.
        box: boolean. Default=True
            If True, select all gas particles that are within a cosmic box 
            centered on the given halo. Otherwise, select halo particles only.
        rewrite: boolean. Default=False
            If True. Reload gas particles from the snapshots even if the files 
            already exist.
        '''
        if(self.haloId is None and haloId is None):
            print("haloId is not found.")
            return
        # Renew haloId if needed
        if(haloId is not None):
            if(self.haloId is None or self.haloId != haloId):
                self.load_halo_info(haloId)
            
        if(box): fname = "box_{:03d}_{:05d}.csv".format(self._snapnum, self.haloId)
        else: fname = "halo_{:03d}_{:05d}.csv".format(self._snapnum, self.haloId)

        fname = os.path.join(self.path_output, fname)
        if(os.path.exists(fname) and not rewrite):
            self._data = pd.read_csv(fname)
        else: # Find particles and write file
            print("Writing particle file {}".format(os.path.basename(fname)))
            if(box):
                xmin, xmax = self.xlims
                ymin, ymax = self.ylims
                zmin, zmax = self.zlims
                self._snap.load_gas_particles(['x','y','z','Mc','Sfr','logT'])
                self._data = self._snap.gp.query("x > @xmin and x < @xmax")
                self._data = self._data.query("y > @ymin and y < @ymax")
                self._data = self._data.query("z > @zmin and z < @zmax")
            else:
                self._snap.load_gas_particles(['x','y','z','Mc','Sfr','logT','haloId'])
                self._data = self._snap.gp.query("haloId==@haloId")
            if(not os.path.isdir(self.path_output)):
                os.mkdir(self.path_output)
            self._data.to_csv(fname, index=False)

    @staticmethod
    def get_color_index(x, cmap, gtype="gas"):
        if(gtype=="gas"):
            val = (x - 4.0) / 2.5
            val = max(0.0, val)
            val = min(1.0, val)
            return cmap(val)
        if(gtype=="sfr"):
            return cmap
        if(gtype=="phew"):
            return cmap(x)
            
    def set_color_and_size(self):
        flag_sfr = (self._data.Sfr > 0)
        flag_phew = (self._data.Mc > 0)
        
        self._data.loc[~(flag_sfr | flag_phew), 'color'] \
            = self._data.loc[~(flag_sfr | flag_phew), 'logT'].apply(
                self.get_color_index, args=(self.cmap_logt, "gas"))
        self._data.loc[~(flag_sfr | flag_phew), 'sizes'] = 5

        self._data.loc[flag_sfr, 'color'] \
            = self._data.loc[flag_sfr, 'Sfr'].apply(
                self.get_color_index, args=(self.cmap_star, "sfr"))
        self._data.loc[flag_sfr, 'sizes'] = 5

        self._data.loc[flag_phew, 'color'] \
            = self._data.loc[flag_phew, 'Mc'].apply(
                self.get_color_index, args=(self.cmap_phew, "phew"))
        self._data.loc[flag_phew, 'sizes'] \
            = 20. * self._data.loc[flag_phew, 'Mc'] ** 2

    def set_canvas(self):
        self.fig = plt.figure(1, figsize=self.figsize)
        self.ax_main = self.fig.add_axes([0.0,0.0,0.665,1.0], projection="3d")
        self.ax_main.set_facecolor(self.background)
        self.ax_main._axis3don = False
        self.ax_edgeon = self.fig.add_axes([0.67,0.0,0.33,0.495], projection="3d")
        self.ax_edgeon.set_facecolor(self.background)
        self.ax_edgeon._axis3don = False    
        self.ax_faceon = self.fig.add_axes([0.67,0.505,0.33,0.495], projection="3d")
        self.ax_faceon.set_facecolor(self.background)
        self.ax_faceon._axis3don = False

    def draw(self, savefig=True, path_figure=None):
        '''
        Draw 3 views of the halo: One overview, another two zoomed-in on the 
        central region. One head-on view, another edge-on.

        Parameters
        ----------
        savefig: boolean. Default = False
            If True. Save the figure as a PNG file.
        path_figure: string. Default = None.
            Path to the figure folder. If None, use default figure path 
            specified in the configuration file.
        '''
        
        print("Set Canvas ...")
        self.set_canvas()
        print("Set color and size ...")
        self.set_color_and_size()
        self.ax_main.scatter(self._data.x, self._data.y, self._data.z,
                             marker='o', c=self._data.color, s=self._data.sizes,
                             edgecolors='none')
        txt = "z = {:3.1f}".format(self._snap.redshift)
        self.ax_main.text2D(0.08, 0.92, txt, fontsize=16, color="lightgrey",
                            weight='heavy', transform=self.ax_main.transAxes)
        self.ax_main.set_xlim(self.x - self.rlim * self.rad,
                              self.x + self.rlim * self.rad)
        self.ax_main.set_ylim(self.y - self.rlim * self.rad,
                              self.y + self.rlim * self.rad)
        self.ax_main.set_zlim(self.z - self.rlim * self.rad,
                              self.z + self.rlim * self.rad)
        # for si in range(len(sizes)):
        #     if(sizes[si] != 3): sizes[si] *= 2.0
        # Face On
        self.ax_faceon.scatter(self._data.x, self._data.y, self._data.z,
                               marker='o', c=self._data.color, s=self._data.sizes,
                               edgecolors='none')
        self.ax_faceon.set_xlim(self.x - self.rlim_zoomin * self.rad,
                                self.x + self.rlim_zoomin * self.rad)
        self.ax_faceon.set_ylim(self.y - self.rlim_zoomin * self.rad,
                                self.y + self.rlim_zoomin * self.rad)
        self.ax_faceon.set_zlim(self.z - self.rlim_zoomin * self.rad,
                                self.z + self.rlim_zoomin * self.rad)
        self.ax_faceon.view_init(self.angles_faceon[0], self.angles_faceon[1])    
        # Edge On
        self.ax_edgeon.scatter(self._data.x, self._data.y, self._data.z,
                               marker='o', c=self._data.color, s=self._data.sizes,
                               edgecolors='none')
        self.ax_edgeon.set_xlim(self.x - self.rlim_zoomin * self.rad,
                                self.x + self.rlim_zoomin * self.rad)
        self.ax_edgeon.set_ylim(self.y - self.rlim_zoomin * self.rad,
                                self.y + self.rlim_zoomin * self.rad)
        self.ax_edgeon.set_zlim(self.z - self.rlim_zoomin * self.rad,
                                self.z + self.rlim_zoomin * self.rad)
        self.ax_edgeon.view_init(self.angles_edgeon[0], self.angles_edgeon[1])

        if(savefig):
            if(path_figure is None):
                path_figure = self.path_figure
            figname = "frame_{:3d}_{:05d}.png".format(self._snapnum, self.haloId)
            figname = os.path.join(path_figure, figname)
            plt.savefig(figname)

    def select_galaxies_by_mass_percentiles(self, plow=0.99, phigh=1.00):
        '''
        Helper function that displays a list of galaxies whose masses fall 
        within the given percentile range.
        '''
        return self._snap.select_galaxies_by_mass_percentiles(plow, phigh)

    def halo(self):
        '''
        Display information of the currently selected halo.
        '''
        if(self.haloId is None):
            print("\nHalo not loaded yet.")
        line = "HaloId: {}".format(self.haloId)
        line += "\n--------------------------------"
        line += "\nCenter  : [{:7.1f}, {:7.1f}, {:7.1f}]".format(
            self.x, self.y, self.z)
        line += "\nlogMvir : {:5.2f}".format(self.mvir)
        line += "\nRvir    : {:5.1f} kpc".format(self.rad)
        print(line)

    def generate_movie_frames(self, path_output=None):
        '''
        Generate movie frames for the currently selected halo at all previous 
        snapshots since it has formed. The frames can be used later for making 
        movies that show the evolution of the halo, and in particular, the 
        winds that constantly come out of the galaxy.

        Parameters
        ----------
        path_output: string. Default = None
            By default, output to the tmpdir.
        '''

        folder = "frames_{:3d}_{:05d}".format(self._snapnum, self.haloId)
        if(path_output is None):
            path_output = os.path.join(self.path_figure, folder)
        if(not os.path.isdir(path_output)):
            os.mkdir(path_output)

        # Load progenitor table to find the progenitors of the halo
        tab = self._snap.load_progtable()
        tab = tab.loc[self.haloId][['snapnum', 'progId']].set_index('snapnum')

        for snapnum in tab.index:
            progid = tab.loc[snapnum]
            figname = "box_{:3d}_{:05d}.png".format(snapnum, progid)
            if(progid == 0): continue
            path_figure = os.path.join(path_output, figname)
            snap = snapshot.Snapshot(self.snapnum, self._snap.model)
            h3d = Halo3D(snap, prog)
            h3d.draw(savefig=True)
            plt.close()

    @property
    def halos(self):
        return self._snap.halos

    @staticmethod
    def demo():
        '''
        Another halo for demonstration is:
        model = l25n288-phew-m5
        snapnum = 58
        haloId = 48
        angles_faceon = (98, 5)
        angles_edgeon = (117, -84)
        '''
        from .. import snapshot
        snap = snapshot.Snapshot('l25n144-test', 108)        
        h3d = Halo3D(snap, haloid=147)
        h3d.draw()
        plt.show()

