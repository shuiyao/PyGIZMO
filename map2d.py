import pandas as pd
import numpy as np
import snapshot
from myinit import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Map2D():
    def __init__(self, snap, ax, zrange=(0.0, 1.0)):
        self._snap = snap
        self.ax = ax
        self.zrange = zrange
        self.boxsize = self._snap.boxsize
        self.zmin = self.zrange[0] * self.boxsize
        self.zmax = self.zrange[1] * self.boxsize
        self.xlims = (0.0, self.boxsize)
        self.ylims = (0.0, self.boxsize)
        # self.xlims = (16000.0, 20000.0)
        # self.ylims = (18000.0, 22000.0)
        self.xticks = np.linspace(self.xlims[0]/1.e3, self.xlims[1]/1.e3, 5)
        self.yticks = np.linspace(self.ylims[0]/1.e3, self.ylims[1]/1.e3, 5)

    def add_density_map(self, ncells=(128, 128), cmap='Purples', seaborn=True):
        self.set_colormap(cmap)
        self._snap.load_gas_particles(['x','y','z','Mass'])
        gp = self._snap.gp

        gp = gp[(gp.z > self.zmin) & (gp.z < self.zmax)]
        gp = gp[(gp.x > self.xlims[0]) & (gp.x < self.xlims[1])]
        gp = gp[(gp.y > self.ylims[0]) & (gp.y < self.ylims[1])]
        ncells_x, ncells_y = ncells[0], ncells[1]
        df = gp.assign(
            xbins=pd.cut(gp.x, ncells_x, labels=range(ncells_x)),
            ybins=pd.cut(gp.y, ncells_y, labels=range(ncells_y))
        )
        self.df = df
        data = df.groupby(['xbins','ybins']).sum('Mass')
        data['Mass'] = data['Mass'].apply(np.log10)
        vmin = data.Mass.replace([-np.inf], 0.0).min()
        vmax = data.Mass.max()
        data = data.reset_index().pivot('ybins', 'xbins', 'Mass')
        if(seaborn):
            hmap = sns.heatmap(data, xticklabels=False, yticklabels=False,
                               cbar=False, vmin=vmin, vmax=vmax, cmap=self.cmap,
                               ax=self.ax)
            self.ax.invert_yaxis()        
        else:
            data = data.to_numpy()
            xbins = np.linspace(self.xlims[0], self.xlims[1], ncells_x)
            ybins = np.linspace(self.xlims[0], self.xlims[1], ncells_y)        
            xgrid, ygrid = np.meshgrid(xbins, ybins)
            self.ax.pcolor(xgrid, ygrid, data, cmap="Purples")

    def add_halos(self, rmin=100.0, show_index=False):
        self._snap.load_halos(['Rvir','x','y','z'])
        halos = self._snap.halos
        halos['x'] = halos['x'].apply(self._snap._transform_coordinates)
        halos['y'] = halos['y'].apply(self._snap._transform_coordinates)
        halos['z'] = halos['z'].apply(self._snap._transform_coordinates)        
        halos = snap.halos[(halos.z > self.zmin) & (halos.z < self.zmax)]
        halos = halos[halos['Rvir'] > rmin]

        print("Adding {} halos to plot...".format(halos.shape[0]))
        for halo in halos.iterrows():
            x, y, r = self._xnorm(halo[1].x), self._ynorm(halo[1].y), self._xnorm(halo[1].Rvir)
            self.ax.add_artist(Circle((x, y), r, alpha=0.6, fc="black"))
            if(show_index==True):
                lbl = "{:d}".format(halo[0])
                self.ax.text(x, y, lbl, color="orange", fontsize=9)

    def set_colormap(self, cmap):
        try:
            self.cmap = plt.get_cmap(cmap)
        except: KeyError

    def draw(self):
        self._set_ticks()
        self.ax.set_xlabel("x [Mpc/h]")
        self.ax.set_ylabel("y [Mpc/h]")
        plt.show()

    def _xnorm(self, x):
        '''
        Normalized value of x onto the x-axis of the Axes ax.
        '''
        l, r = self.ax.get_xlim()[0], self.ax.get_xlim()[1]
        v = (x - self.xlims[0]) / (self.xlims[1] - self.xlims[0])
        return l + v * (r - l)

    def _ynorm(self, y):
        '''
        Normalized value of y onto the y-axis of the Axes ax.
        '''
        l, r = self.ax.get_ylim()[0], self.ax.get_ylim()[1]
        v = (y - self.ylims[0]) / (self.ylims[1] - self.ylims[0])
        return l + v * (r - l)

    def _set_ticks(self):
        self.ax.set_xticks([self._xnorm(x*1000.) for x in self.xticks])
        labels = ["{:4.1f}".format(x) for x in self.xticks]
        self.ax.set_xticklabels(labels)
        
        self.ax.set_yticks([self._ynorm(y*1000.) for y in self.yticks])
        labels = ["{:4.1f}".format(y) for y in self.yticks]        
        self.ax.set_yticklabels(labels)
        self.ax.tick_params(direction="in")

reload(snapshot)
model = "l25n144-test"
snap = snapshot.Snapshot(model, 108)
fig, ax = plt.subplots(1, 1, figsize=(8,8))
# map2d = Map2D(snap, ax, zrange=(0.3, 0.5))
map2d = Map2D(snap, ax)
map2d.add_density_map()
map2d.add_halos(show_index=True)
map2d.draw()
