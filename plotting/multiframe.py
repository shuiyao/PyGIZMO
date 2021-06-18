__all__ = ['FrameTopLevel', 'FrameMulti']

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import parameters
import numpy as np
from importlib import reload

# mpl.rcParams['text.usetex'] = True

class FrameTopLevel():
    def __init__(self, nrows, ncols):
        # self._params = parameters.ParamsTopLevel(nrows, ncols)

        self.dims = (nrows, ncols)
        self.logscale_x = False
        self.logscale_y = False
        self.np = nrows * ncols

        self.axisON = [True] * self.np
        self.xlims = [()] * self.np
        self.ylims = [()] * self.np
        self.xlabels = [""] * self.np
        self.ylabels = [""] * self.np
        self.xticks = [[]] * self.np
        self.yticks = [[]] * self.np
        self.xticklabels = [[]] * self.np
        self.yticklabels = [[]] * self.np
        self.xticksON = [True] * self.np
        self.yticksON = [True] * self.np
        self.xtickformat = ["%3.1f"] * self.np
        self.ytickformat = ["%3.1f"] * self.np
        self.xrotation = [0] * self.np # label rotation
        self.yrotation = [0] * self.np
        for i in range(self.np): self.xlabels[i] = ""
        for i in range(self.np): self.ylabels[i] = ""

        # TODO:

    @staticmethod
    def _index(row, col, nrows=1, ncols=1):
        if(row < 0): row = nrows + row
        if(col < 0): col = ncols + col
        return row * ncols + col

    def gen_ticklabels(self, ticks, tickformat):
        ''' Convert number ticks into strings '''
        ticklabels = []
        for tick in ticks:
            ticklabels.append(tickformat % (tick))
        return ticklabels

    '''
    which:
        string: One of the following ['all', 'row', 'col', 'column', 'left', 
                'bottom', 'right', 'top', 'none']
        integer: Only change the panel with the index equals to which
        list/tuple: Must be two elements only, specifying the row(s) and col(s)
                    If like (2,3), fix the panel with row=2, col=3
                    If like ([1,2], [2,3]), fix panels (1,2) and (2,3)

    Examples
    --------
    nrow = 2, ncol = 3:
    | 0 1 2 |
    | 3 4 5 |

    >>> FrameTopLevel._parse_index('all', 2, 3)
    [0, 1, 2, 3, 4, 5]
    >>> FrameTopLevel._parse_index('row', 2, 3)
    [3, 4, 5]
    >>> FrameTopLevel._parse_index('col', 2, 3)
    [0, 3]
    >>> FrameTopLevel._parse_index('right', 2, 3)
    [2, 5]
    >>> FrameTopLevel._parse_index('top', 2, 3)
    [0, 1, 2]
    >>> FrameTopLevel._parse_index([1,2], 2, 3)
    [5]
    >>> FrameTopLevel._parse_index([(0,1),(0,2),(1,2)], 2, 3)
    [1, 2, 5]
    '''

    @staticmethod
    def _parse_index(which, nrows, ncols):
        np = nrows * ncols
        if(isinstance(which, str)):
            if(which == 'all'): return [i for i in range(np)]
            if(which == 'none'): return []
            if(which in ['row','bottom']):
                return [FrameTopLevel._index(-1, col, nrows, ncols)
                        for col in range(ncols)]
            if(which in ['col','left','column']):
                return [FrameTopLevel._index(row, 0, nrows, ncols)
                        for row in range(nrows)]
            if(which == 'top'):
                return [FrameTopLevel._index(0, col, nrows, ncols)
                        for col in range(ncols)]
            if(which == 'right'):
                return [FrameTopLevel._index(row, -1, nrows, ncols)
                        for row in range(nrows)]
            
        if(isinstance(which, int)):
            # 1 -> [1]
            return [which]

        if(isinstance(which, list) or isinstance(which, tuple)):
            # If erase=True, this sets every panel to default
            if(len(which) == 0): return []
            if(isinstance(which[0], int)):
                assert(isinstance(which[1], int)), "The two elements in the 'which' argument must have the same form (int or list/tuple with same length)."
                # [1,2] -> idx(1,2)
                return [FrameTopLevel._index(which[0], which[1], ncols=ncols)]
            return [FrameTopLevel._index(which[i][0], which[i][1], ncols=ncols)
                    for i in range(len(which))]

    @staticmethod
    def set_panels_attribute(pattr, val, dims, which='all', erase=False, default_val=None):
        '''
        pattrs: 
            A list of size equals to the total number of panels in the frame.
        '''
        np = dims[0] * dims[1]
        idx = FrameTopLevel._parse_index(which, dims[0], dims[1])
        for i in idx: pattr[i] = val
        if(erase):
            assert(default_val is not None), "default_val is required when erase is True."
            for i in set(idx) ^ set(range(np)):
                pattr[i] = default_val

        # x properties

    def set_xlabels(self, xlabel, which='all', erase=False):
        ''' Set xlabels for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xlabels, xlabel, self.dims, default_val="", which=which, erase=erase)

    def set_xticks(self, xticks, which='all', erase=False):
        ''' Set xticks for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xticks, xticks, self.dims, default_val=[], which=which, erase=erase)
        
    def set_xticklabels(self, xticklabels, which='all', erase=False):
        ''' Set xticklabels for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xticklabels, xticklabels, self.dims, default_val=[], which=which, erase=erase)
        
    def set_xtickformat(self, xtickformat, which='all', erase=False):
        ''' Set xtickformats for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xtickformats, xtickformat, self.dims, default_val=[], which=which, erase=erase)

    def set_xlims(self, xmin, xmax, which='all', erase=False):
        ''' Set xlims for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xlims, (xmin, xmax), self.dims, default_val=(), which=which, erase=erase)

    def set_xrotation(self, xrotation, which='all', erase=False):
        ''' Set xrotation for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.xrotation, xrotation, self.dims, default_val=0, which=which, erase=erase)
        
        # y properties

    def set_ylabels(self, ylabel, which='all', erase=False):
        ''' Set ylabels for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.ylabels, ylabel, self.dims, default_val="", which=which, erase=erase)

    def set_yticks(self, yticks, which='all', erase=False):
        ''' Set yticks for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.yticks, yticks, self.dims, default_val=[], which=which, erase=erase)
        
    def set_yticklabels(self, yticklabels, which='all', erase=False):
        ''' Set yticklabels for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.yticklabels, yticklabels, self.dims, default_val=[], which=which, erase=erase)

    def set_ytickformat(self, ytickformat, which='all', erase=False):
        ''' Set ytickformats for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.ytickformats, ytickformat, self.dims, default_val=[], which=which, erase=erase)

    def set_ylims(self, ymin, ymax, which='all', erase=False):
        ''' Set ylims for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.ylims, (ymin, ymax), self.dims, default_val=(), which=which, erase=erase)

    def set_yrotation(self, yrotation, which='all', erase=False):
        ''' Set yrotation for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.yrotation, yrotation, self.dims, default_val=0, which=which, erase=erase)

    def set_fontsize(self, fontsize, which):
        if(which == 'label'):
            self._params.fontsize_label = fontsize
        elif(which == 'tick'):
            self._params.fontsize_tick = fontsize            

    @staticmethod
    def _print_params_panels(key, vals, nrows, ncols):
        k = 0
        for i in range(nrows):
            line = "{:16s}: ".format(key) if i == 0 else "{:16s}  ".format("")
            for j in range(ncols):
                line += "{:s}, ".format(str(vals[k]))
                k = k + 1
            print(line)
        
    def parameters(self, level="all"):
        '''
        Display parameters at a specified level.
        '''
        nr, nc = self.dims[0], self.dims[1]
        self._params.show()
        if(level == 'all'):
            self._print_params_panels("xlims", self.xlims, nr, nc)
            self._print_params_panels("ylims", self.ylims, nr, nc)
            self._print_params_panels("xlabels", self.xlabels, nr, nc)
            self._print_params_panels("ylabels", self.ylabels, nr, nc)
            self._print_params_panels("xticksON", self.xticksON, nr, nc)
            self._print_params_panels("yticksON", self.yticksON, nr, nc)
            self._print_params_panels("xrotation", self.xrotation, nr, nc)
            self._print_params_panels("yrotation", self.yrotation, nr, nc)
        
    def man(self):
        show_doc("doc/panels.man")
        
    def help(self):
        self.man()

    def draw(self, verbose=False, sketch=False):
        # Figure Level
        pars = self._params
        fig = plt.figure(pars.num, figsize=pars.figsize)

        # Frame Level
        axs = []
        gs = gridspec.GridSpec(
            pars.dims[0], pars.dims[1],
            height_ratios = pars.height_ratios, width_ratios = pars.width_ratios)
        for i in range(pars.npanels):
            axs.append(plt.subplot(gs[i]))

        # Ax Level
        for i in range(pars.npanels):
            # ranges
            if(len(self.xlims[i]) != 0):
                axs[i].set_xlim(self.xlims[i])
            if(len(self.ylims[i]) != 0):
                axs[i].set_ylim(self.ylims[i])
            # ticklabels
            if(len(self.xticks[i]) != 0):
                axs[i].set_xticks(self.xticks[i])
                if(len(self.xticklabels[i]) == 0):
                    axs[i].set_xticklabels(
                        self.gen_ticklabels(self.xticks[i], self.xtickformat[i]),
                        fontsize=self._params.fontsize_tick)
                else:
                    axs[i].set_xticklabels(
                        self.xticklabels[i],
                        fontsize=self._params.fontsize_tick)
                    
            if(len(self.yticks[i]) != 0):
                axs[i].set_yticks(self.yticks[i])
                if(len(self.yticklabels[i]) == 0):
                    axs[i].set_yticklabels(
                        self.gen_ticklabels(self.yticks[i], self.ytickformat[i]),
                        fontsize=self._params.fontsize_tick)
                else: axs[i].set_yticklabels(
                        self.yticklabels[i],
                        fontsize=self._params.fontsize_tick)
            if(not self.xticksON[i]):
                axs[i].set_xticklabels([])
            if(not self.yticksON[i]):
                axs[i].set_yticklabels([])
            # labels
            if(self.xticksON[i]):
                axs[i].set_xlabel(r'{}'.format(self.xlabels[i]),
                                  fontsize=self._params.fontsize_label)
            if(self.yticksON[i]):
                axs[i].set_ylabel(r'{}'.format(self.ylabels[i]),
                                  fontsize=self._params.fontsize_label)
            # logscale
            if(self.logscale_x == True): axs[i].set_xscale("log")
            if(self.logscale_y == True): axs[i].set_yscale("log")

            if(not self.axisON[i]):
                axs[i].axis('off')
            if(sketch == True):
                axs[i].text(0.9, 0.9, str(i), transform = axs[i].transAxes)

        fig.subplots_adjust(hspace=pars.hspace, wspace=pars.wspace, \
                            bottom=pars.bottom, top=pars.top, left=pars.left, right=pars.right)
        fig.suptitle(pars.title)
        if(verbose == True):
            pars.show() # Show parameters
        return fig, axs

    def sketch(self):
        self.draw(verbose=True, sketch=True)

    def avail():
        pass
    
    def help():
        pass

class FrameMulti(FrameTopLevel):
    def __init__(self, nrows, ncols, tight_layout=True):
        super(FrameMulti, self).__init__(nrows, ncols)
        self._params = parameters.ParamsMulti(nrows, ncols, tight_layout)
        if(tight_layout):
            self.xticksON = [False] * self.np
            self.yticksON = [False] * self.np
            for i in range(self.dims[1]):
                self.xticksON[-i-1] = True
            for i in range(self.dims[0]):
                self.yticksON[self.dims[1]*i] = True
        else:
            self.xticksON = [True] * self.np
            self.yticksON = [True] * self.np

#   +-------+-------+
#   |       |       |
# y |       |       |
#   |       |       |
#   +-------+-------+
#   |       |       |
# y |       |       |
#   |       |       |
#   +-------+-------+
#       x       x

frm = FrameMulti(2,2,False)
frm._params.hspace=0.25
frm.set_xlabels('x', which='row')
frm.set_ylabels('y', which='col')

#   +-------+---+
#   |       |   |
# y1|       |   |
#   |       |   |
#   +-------+---+
# y2|       | x
#   +-------+
#       x

# frm = FrameMulti(2,2)
# frm._params.height_ratios = [4, 1]
# frm._params.width_ratios = [4, 1]
# frm.set_xlabels('x', which=[(1,0),(0,1)])
# frm.set_ylabels('y1', which=(0,0))
# frm.set_ylabels('y2', which=(1,0))
# frm.axisON[3] = False

#   +-------+-------+
#   |       |       |
# y1|       |       |
#   |       |       |
#   |       |       |
#   +-------+-------+
# y2|       |       |
#   +-------+-------+
#       x       x

# frm = FrameMulti(2,2)
# frm._params.height_ratios = [4, 1]
# frm.set_ylabels('y1', which=(0,0))
# frm.set_ylabels('y2', which=(1,0))


frm.sketch()
plt.show()
