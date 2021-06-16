import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from myinit import *
import snapshot
import os

class Halo3D(object):
    def __init__(self, snap, haloid=None, angles_faceon=[0,0], angles_edgeon=[0,0], rlim=0.8, rlim_zoomin=0.15, cmap_logt="plasma", cmap_phew="Greens", background="black", figsize=(9,6)):
        self._snap = snap
        self._snapnum = snap.snapnum
        self.rlim = rlim
        self.rlim_zoomin = rlim_zoomin
        self.angles_faceon = angles_faceon
        self.angles_edgeon = angles_edgeon
        self.path_output = snap._path_tmpdir
        self.haloId = haloid
        if(haloid is not None):
            self.load_halo_info(haloid)
            self.load_halo_particles()

        self.cmap_logt=plt.get_cmap(cmap_logt) # 4.0 - 7.0
        self.cmap_phew=plt.get_cmap(cmap_phew) # 0.0 - 1.0
        self.color_star = (65./255., 105./255., 225./255., 1.0) # royal blue
        self.figsize = figsize
        self.background = background

    def load_halo_info(self, haloId):
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
        if(self.haloId is None and haloId is None):
            print("haloId is not found.")
            return
        # Renew haloId if needed
        if(haloId is not None):
            if(self.haloId is None or self.haloId != haloId):
                self.load_halo_info(haloId)
            
        if(box): fname = "box_{:3d}_{:5d}.csv".format(self._snapnum, self.haloId)
        else: fname = "halo_{:3d}_{:5d}.csv".format(self._snapnum, self.haloId)
        fname = os.path.join(self.path_output, fname)
        if(os.path.exists(fname) and not rewrite):
            self._data = pd.read_csv(fname)
        else: # Find particles and write file
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

        gp = self._data[~(flag_sfr | flag_phew)]
        gp['color'] = gp.logT.apply(
            self.get_color_index, args=(self.cmap_logt, "gas"))
        gp['sizes'] = 3

        ism = self._data[flag_sfr]
        ism['color'] = ism.Sfr.apply(
            self.get_color_index, args=(self.color_star, "sfr"))
        ism['sizes'] = 3
        
        phew = self._data[flag_phew]
        phew['color'] = phew.Mc.apply(
            self.get_color_index, args=(self.cmap_phew, "phew"))
        phew['sizes'] = 20. * phew.Mc ** 2

        self._data = pd.concat([gp, ism, phew], axis=0)

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

    def draw(self):
        '''
        Draw 3 views of the halo: One overview, another two zoomed-in on the 
        central region. One head-on view, another edge-on.
        '''
        print("Set Canvas ...")
        self.set_canvas()
        print("Set color and size ...")
        self.set_color_and_size()
        self.ax_main.scatter(self._data.x, self._data.y, self._data.z,
                             marker='o', c=self._data.color, s=self._data.sizes,
                             edgecolors='none')
        txt = "z = {:3.1f}".format(snap.redshift)
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

    def halo(self):
        if(self.haloId is None):
            print("\nHalo not loaded yet.")
        line = "HaloId: {}".format(self.haloId)
        line += "\n--------------------------------"
        line += "\nCenter  : [{:7.1f}, {:7.1f}, {:7.1f}]".format(
            self.x, self.y, self.z)
        line += "\nlogMvir : {:5.2f}".format(self.mvir)
        line += "\nRvir    : {:5.1f} kpc".format(self.rad)
        print(line)

    @property
    def halos(self):
        return self._snap.halos

h3d = Halo3D(snap, haloid=1)
h3d.draw()
plt.savefig(DIRS['FIGURE']+"tmp.png")

