import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pdb import set_trace

import snapshot

class Map2D(object):
    def __init__(self, snap, ax, zrange=(0.0, 1.0), xlims=(0.0, 1.0), ylims=(0.0, 1.0)):
        self._snap = snap
        self.ax = ax
        self.zrange = zrange
        self.boxsize = self._snap.boxsize
        self.zmin = self.zrange[0] * self.boxsize
        self.zmax = self.zrange[1] * self.boxsize
        self.xlims = (xlims[0] * self.boxsize, xlims[1] * self.boxsize)
        self.ylims = (ylims[0] * self.boxsize, ylims[1] * self.boxsize)        
        self.xticks = np.linspace(self.xlims[0]/1.e3, self.xlims[1]/1.e3, 5)
        self.yticks = np.linspace(self.ylims[0]/1.e3, self.ylims[1]/1.e3, 5)

    def add_layer_density_map(self, ncells=(128, 128), cmap=None, seaborn=True, layer="density"):
        if(cmap is None):
            if(layer == "density"): cmap = "Purples"
            if(layer == "temperature"): cmap = "rocket"
        
        self.set_colormap(cmap)
        self._snap.load_gas_particles(['x','y','z','Mass'])
        if(layer == "temperature"):
            self._snap.load_gas_particles(['logT'], drop=False)
            
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

        if(layer == "density"):
            data = df.groupby(['xbins','ybins']).sum('Mass')
            data['Mass'] = data['Mass'].apply(np.log10)
            vmin = data.Mass.replace([-np.inf], 0.0).min()
            vmax = data.Mass.max()
            data = data.reset_index().pivot('ybins', 'xbins', 'Mass')
            cbar = False
        elif(layer == "temperature"):
            df['mlogT'] = df['Mass'] * df['logT']
            data = df.groupby(['xbins','ybins']).sum()
            data['logT'] = data['mlogT'] / data['Mass']
            # data.logT.replace([-np.inf, np.inf], 0.0, inplace=True)
            data.logT.fillna(0.0, inplace=True)
            vmin = 3.0
            vmax = 8.0
            data = data.reset_index().pivot('ybins', 'xbins', 'logT')
            cbar = True
            cbarlabel = r"$log(T/K)$"
            
        if(seaborn):
            hmap = sns.heatmap(data, xticklabels=False, yticklabels=False,
                               ax=self.ax, vmin=vmin, vmax=vmax, cmap=self.cmap,
                               cbar=cbar, cbar_kws={'label': cbarlabel})
            self.ax.invert_yaxis()        
        else:
            data = data.to_numpy()
            xbins = np.linspace(self.xlims[0], self.xlims[1], ncells_x)
            ybins = np.linspace(self.xlims[0], self.xlims[1], ncells_y)        
            xgrid, ygrid = np.meshgrid(xbins, ybins)
            self.ax.pcolor(xgrid, ygrid, data, cmap="Purples")

    def add_layer_halos(self, rmin=100.0, show_index=False):
        self._snap.load_halos(['Rvir','x','y','z'])
        halos = self._snap.halos
        halos['x'] = halos['x'].apply(self._snap._transform_coordinates)
        halos['y'] = halos['y'].apply(self._snap._transform_coordinates)
        halos['z'] = halos['z'].apply(self._snap._transform_coordinates)        
        halos = snap.halos[(halos.z > self.zmin) & (halos.z < self.zmax)]
        halos = halos[halos['Rvir'] > rmin]

        print("Adding {} halos to plot...".format(halos.shape[0]))
        for halo in halos.iterrows():
            x, y, r = self._xnorm(halo[1].x), self._ynorm(halo[1].y), self._rnorm(halo[1].Rvir)
            self.ax.add_artist(Circle((x, y), r, alpha=0.6, fc="black"))
            if(show_index==True):
                lbl = "{:d}".format(halo[0])
                self.ax.text(x, y, lbl, color="orange", fontsize=9)

    def add_layer_particles(self, cmap=None, skip=None, verbose=False):
        '''
        Add a layer of wind particles to the 2D map.
        '''
        self._snap.load_gas_particles(['Mc', 'Rc'], drop=False)
        gp = self._snap.gp[self._snap.gp.Mc > 0]
        gp = gp[(gp.z > self.zmin) & (gp.z < self.zmax)]
        if(self.xlims[0] > 0):
            gp = gp[(gp.x > self.xlims[0]) & (gp.x < self.xlims[1])]
        if(self.ylims[0] > 0):
            gp = gp[(gp.y > self.ylims[0]) & (gp.y < self.ylims[1])]
        print("Adding a layer of {} wind particles.".format(gp.shape[0]))

        gp.x = gp.x.apply(self._xnorm)
        gp.y = gp.y.apply(self._ynorm)
        cmap = plt.get_cmap('YlGn')
        gp.Rc = gp.Rc * np.sqrt(gp.Mass * 1.e10 / self._snap._h / 1.e5 / gp.Mc)
        gp.Rc = gp.Rc.apply(self._rnorm)

        cnt = 0
        for p in gp.iterrows():
            cnt = cnt + 1
            if(verbose):
                if(cnt % 10000 == 0): print(cnt)
            if(skip is not None):
                if(cnt % skip != 0): continue
            x, y, r, Mc = p[1].x, p[1].y, p[1].Rc, p[1].Mc
            self.ax.add_artist(Circle((x, y), r, alpha=0.4,
                                      fc=cmap(Mc), ec=None))
        
    def set_colormap(self, cmap):
        try:
            self.cmap = plt.get_cmap(cmap)
        except: KeyError

    def draw(self, save=False):
        self._set_ticks()
        self.ax.set_xlabel("x [Mpc/h]")
        self.ax.set_ylabel("y [Mpc/h]")
        if(save):
            plt.savefig("out.png")
        else:
            plt.show()

    def _rnorm(self, rad):
        axnorm = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        return rad * (axnorm / (self.xlims[1] - self.xlims[0]))

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

    # @staticmethod
    # def demo():
    #     from .. import snapshot
# model = "l25n144-test"
# snap = snapshot.Snapshot(model, 108)
# fig, ax = plt.subplots(1, 1, figsize=(8,8))
# map2d = Map2D(snap, ax)
# map2d.add_layer_density_map(layer='temperature', ncells=(128, 128))
# map2d.add_layer_particles()
# map2d.draw()


model = "l25n288-phew-m5"
snap = snapshot.Snapshot(model, 108)
fig, ax = plt.subplots(1, 1, figsize=(8,8))
map2d = Map2D(snap, ax, zrange=(0.3, 0.5))
# map2d = Map2D(snap, ax, zrange=(0.0, 1.0), xlims=(0.45, 0.6), ylims=(0.45, 0.6))
map2d.add_layer_density_map(layer='temperature', ncells=(512, 512))
map2d.add_layer_particles(verbose=True, skip=None)
map2d.draw(save=True)
