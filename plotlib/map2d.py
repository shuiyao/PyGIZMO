import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from scipy import ndimage

from pdb import set_trace

import snapshot

class Map2D(object):

    def __init__(self, snap, ax, zrange=(0.0, 1.0), xlims=(0.0, 1.0), ylims=(0.0, 1.0)):
        self._snap = snap
        self.ax = ax
        self.zrange = zrange

    def set_colormap(self, cmap):
        try:
            self.cmap = plt.get_cmap(cmap)
        except: KeyError

    def draw(self, save=False):
        self._set_ticks()
        self.ax.set_xlabel(r'{}'.format(self.xlabel))
        self.ax.set_ylabel(r'{}'.format(self.ylabel))
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

    def _textnorm(self, x, y, txt, **kwargs):
        '''
        Obtain normalized x, y values and call plt.text() function
        '''
        x = self._xnorm(x)
        y = self._ynorm(y)
        self.ax.text(x, y, txt, **kwargs)

    def _set_ticks(self):
        self.ax.set_xticks([self._xnorm(x*1000.) for x in self.xticks])
        labels = ["{:4.1f}".format(x) for x in self.xticks]
        self.ax.set_xticklabels(labels)
        
        self.ax.set_yticks([self._ynorm(y*1000.) for y in self.yticks])
        labels = ["{:4.1f}".format(y) for y in self.yticks]        
        self.ax.set_yticklabels(labels)
        self.ax.tick_params(direction="in")


class DensityMap(Map2D):
    '''
    Example
    -------
    Make a temperature map over the entire simulation domain, and overplot 
    PhEW particles showing the physical sizes.

    >>> model = "l25n144-test"
    >>> snap = snapshot.Snapshot(model, 108)
    >>> fig, ax = plt.subplots(1, 1, figsize=(8,8))
    >>> map2d = Map2D(snap, ax, zrange=(0.3, 0.6))
    >>> map2d.add_layer_density_map(layer='temperature', ncells=(128, 128))
    >>> map2d.add_layer_particles(verbose=True, skip=None)
    >>> map2d.draw()
    '''
    
    def __init__(self, snap, ax, **kwargs):
        super(DensityMap, self).__init__(snap, ax, **kwargs)

        self.boxsize = self._snap.boxsize
        self.zmin = self.zrange[0] * self.boxsize
        self.zmax = self.zrange[1] * self.boxsize
        self.xlims = (xlims[0] * self.boxsize, xlims[1] * self.boxsize)
        self.ylims = (ylims[0] * self.boxsize, ylims[1] * self.boxsize)        
        self.xticks = np.linspace(self.xlims[0]/1.e3, self.xlims[1]/1.e3, 5)
        self.yticks = np.linspace(self.ylims[0]/1.e3, self.ylims[1]/1.e3, 5)
        self.xlabel = "x [Mpc/h]"
        self.ylabel = "y [Mpc/h]"

        rc_phase_diagram = {
            "font.size":12,
            "axes.titlesize":12,
            "axes.labelsize":8,
            "ytick.labelsize":8,
            "ytick.major.size":2.5,
            "ytick.major.width":0.8,
            "xtick.labelsize":8,
            "xtick.major.size":2.5,
            "xtick.major.width":0.8,
            "ytick.direction":'in',
            "xtick:direction":'in',
            "axes.spines.top":True,
            "axes.spines.right":True
        }

        sns.set_context("paper", rc=rc_phase_diagram)

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
        
class PhaseDiagram(Map2D):
    def __init__(self, snap, ax, cmap='Purples', **kwargs):
        super(DensityMap, self).__init__(snap, ax, **kwargs)

        self.set_colormap(cmap)
        self.xlims = (-1.5, 7.0)
        self.xticks = [-1.0, 1.0, 3.0, 5.0, 7.0]
        self.ylims = (3.0, 8.0)
        self.yticks = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.xlabel = "$log(\rho/\bar{\rho})$"
        self.ylabel = "$log(T/K)$"

    def load_grid_data(self, path_grid):
        rhot = pd.read_csv(path_grid)
        rhot['LogM'] = rhot['Mass'].apply(np.log10)
        self.vmin = rhot['LogM'].replace([-np.inf, np.inf], 0.0).min()
        self.vmax = rhot['LogM'].max()
        print("vmin, vmax = {}, {}".format(self.vmin, self.vmax))
        self.data = rhot.pivot('LogT', 'LogRho', 'LogM')

    def _draw_heatmap(self):
        with sns.axes_style('ticks'):
            hmap = sns.heatmap(self.data, xticklabels=False, yticklabels=False,
                               cbar=False, vmin=self.vmin, vmax=self.vmax,
                               cmap=self.cmap, ax=self.ax)
        sns.despine(right=False, top=False)
        # Reverse the logT order
        self.ax.invert_yaxis()
        self._set_ticks()
    def _nh(x):
        '''
        Convert physical density to cosmic over-density
        '''
        rho_crit_baryon = 1.879e-29 * 0.7 * 0.7 * 0.044
        mh = 1.6733e-24
        return np.log10(10**x * rho_crit_baryon * 0.76 / mh)

    def _rho_thresh(self, z, Om=0.3, OL=0.7):
        f_omega = Om * (1+z)**3
        f_omega = f_omega / (f_omega + OL)
        return 6.*np.pi*np.pi*(1. + 0.4093*(1./f_omega-1.)**0.9052) - 1.

    def _update_top_x_axis(self):
        ax2 = self.ax.twiny()
        x1, x2 = self.xlims[0], self.xlims[1]
        ax2.set_xlim(nh(x1), nh(x2))
        ax2.figure.canvas.draw()
        ax2.set_xlabel(r'$Log(n_H) [cm^{-3}]$')
        ax2.tick_params(direction='in')
        return ax2

    def _set_ticks(self):
        self.ax.set_xticks([self._xnorm(x) for x in self.xticks])
        labels = ["{:-3.1f}".format(x) for x in self.xticks]
        self.ax.set_xticklabels(labels)
        self.ax.set_yticks([self._ynorm(y) for y in self.yticks])
        labels = ["{:-3.1f}".format(y) for y in self.yticks]        
        self.ax.set_yticklabels(labels)
        self.ax.tick_params(direction="in")
   
    def add_particles(self, particles):
        '''
        TODO: overplot particles to the phase diagram
        '''
        pass

    def annotate_phase_regions(self, redshift):
        rhoth = np.log10(self._rho_thresh(redshift)+1.)
        print ("Density Thresh: ", rhoth)
        self.ax.axhline(self._ynorm(5.0), linestyle=":", color="black")
        self.ax.axvline(self._xnorm(rhoth), linestyle=":", color="black")
        self._textnorm(-1.,7.5, "WHIM", color="green")
        self._textnorm(6.,7.5, "Hot", color="red")
        self._textnorm(-1.,4.6, "Diffuse", color="purple")
        self._textnorm(5.2,4.6, "Condensed", color="teal")

    @staticmethod
    def demo():
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        ax = axs[0] if isinstance(axs, list) else axs
        rhot = PhaseDiagram(ax)
        rhot.load_grid_data(path_grid)
        rhot.draw(annotate=True)        
        plt.show()
