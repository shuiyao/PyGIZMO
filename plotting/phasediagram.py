import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

mpl.rcParams["mathtext.default"] = "tt"

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

fig, axs = plt.subplots(1, 1, figsize=(6,6))
ax = axs[0] if isinstance(axs, list) else axs

class PhaseDiagram():
    def __init__(self, ax, cmap="Purples"):
        self.ax = ax
        self.set_colormap(cmap)
        self.xlims = (-1.5, 7.0)
        self.xticks = [-1.0, 1.0, 3.0, 5.0, 7.0]
        self.ylims = (3.0, 8.0)
        self.yticks = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def draw(self, annotate=True):
        self._draw_heatmap()
        ax_top = self._update_top_x_axis()
        self.ax.set_xlabel(r'$log(\rho/\bar{\rho})$')        
        self.ax.set_ylabel(r'$log(T/K)$')
        if(annotate):
            self.annotate_phase_regions(0.0)

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

    def set_colormap(self, cmap):
        try:
            self.cmap = plt.get_cmap(cmap)
        except: KeyError

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
        self.ax.set_xticklabels(["-1.0", "1.0", "3.0", "5.0", "7.0"])
        self.ax.set_yticks([self._ynorm(y) for y in self.yticks])
        self.ax.set_yticklabels(["3.0", "4.0", "5.0", "6.0", "7.0", "8.0"])
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

rhot = PhaseDiagram(ax)
rhot.load_grid_data(path_grid)
rhot.draw(annotate=True)        
#plt.show()

# grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
# f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
# ax = sns.heatmap(flights, ax=ax,
#                  cbar_ax=cbar_ax,
#                  cbar_kws={"orientation": "horizontal"})

print("done")
