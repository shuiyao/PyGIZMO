__all__ = ['FrameTopLevel', 'FrameMulti']

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from importlib import reload
from pdb import set_trace

from . import parameters
from .legend import Legend

# mpl.rcParams['text.usetex'] = True

def show_doc(fname):
    fname = os.path.join(fbase, fname)
    print ("Document: %s" % (fname))
    print ("----------------------------------------------------------------")
    f = open(fname, "r")
    print (f.read())
    f.close()

class FrameTopLevel(object):
    '''
    General class for setting up the layout of a figure.
    '''
    
    def __init__(self, nrows, ncols):
        # self._params = parameters.ParamsTopLevel(nrows, ncols)

        self.dims = (nrows, ncols)
        self.logscale_x = False
        self.logscale_y = False
        self.np = nrows * ncols
        self.legends = []

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
        self.xtickformat = ["%3.1g"] * self.np
        self.ytickformat = ["%3.1g"] * self.np
        self.xrotation = [0] * self.np # label rotation
        self.yrotation = [0] * self.np
        for i in range(self.np): self.xlabels[i] = ""
        for i in range(self.np): self.ylabels[i] = ""

    @classmethod
    def from_config(cls, path_config=None):
        '''
        Read parameters from a user defined configuration file.
        TODO
        '''
        if(path_config is None):
            path_dir = os.path.dirname(os.path.abspath(__file__))
            path_config = os.path.join(path_dir, "multiframe.cfg")
        if(not os.path.exists(path_config)):
            raise IOError('Config file {} not found.'.format(path_config))
        return cls
        

    def gen_ticklabels(self, ticks, tickformat):
        ''' Convert numeric ticks into strings '''
        ticklabels = []
        for tick in ticks:
            ticklabels.append(tickformat % (tick))
        return ticklabels

    @staticmethod
    def _index(row, col, nrows=1, ncols=1):
        if(row < 0): row = nrows + row
        if(col < 0): col = ncols + col
        return row * ncols + col

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
                return [FrameTopLevel._index(which[0], which[1], nrows, ncols)]
            return [FrameTopLevel._index(which[i][0], which[i][1], nrows, ncols)
                    for i in range(len(which))]

    def set_param(self, par, value=None):
        '''
        Set figure level parameter.

        Parameters
        ----------
        par: string or dict.
            The parameter to set.
            If string, value must not be None.

        value: var. Default = None.
            The new value
            If None, assuming par is a dictionary.
        '''

        if(isinstance(par, str)):
            par = {par:value}
        assert(isinstance(par, dict)), "set_param(): par must be either a string of a dictionary."
        for key in par.keys():
            if(key in self._params.__dict__.keys()):
                setattr(self._params, key, par[key])                
            else:
                print("{} is not found in the parameter list, ignore.".format(key))

    @staticmethod
    def set_panels_attribute(pattr, val, dims, which='all', erase=False, default_val=None):
        '''
        Set some attribute of selected panels in a figure to a specific value.

        Parameters
        ----------
        pattrs: var
            The panel attribute that is to reset.

        val: var
            The value to set to the selected panels.

        dims: list/tuple of size 2.
            (number_of_rows, number_of_columns)

        which: int, string or list/tuple.
            string: One of the following ['all', 'row', 'col', 'column', 'left', 
                    'bottom', 'right', 'top', 'none']
            integer: Only change the panel with the index equals to which
            list/tuple: Must be two elements only, specifying the row(s) and col(s)
                        If like (2,3), fix the panel with row=2, col=3
                        If like ([1,2], [2,3]), fix panels (1,2) and (2,3)

            Some examples:
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

        erase: boolean. Default = False.
            If True, reset the attribute of all panels that are NOT selected 
            to default.

        default_val: var. Default = None.
            The default value for the attribute.

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
        ''' 
        Set xlabels for 'which' panels. 
        See help(frm.set_panels_attribtes) for more details
        '''
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
            self.xtickformat, xtickformat, self.dims, default_val=[], which=which, erase=erase)

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
            self.ytickformat, ytickformat, self.dims, default_val=[], which=which, erase=erase)

    def set_ylims(self, ymin, ymax, which='all', erase=False):
        ''' Set ylims for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.ylims, (ymin, ymax), self.dims, default_val=(), which=which, erase=erase)

    def set_yrotation(self, yrotation, which='all', erase=False):
        ''' Set yrotation for 'which' panels. '''
        FrameTopLevel.set_panels_attribute(
            self.yrotation, yrotation, self.dims, default_val=0, which=which, erase=erase)

    def set_fontsize(self, which, fontsize):
        '''
        Set the fontsize for figure objects.

        Parameters
        ----------
        which: int, 2-tuple or string
            Determines the index of the axes to add the legend.

        fontsize: int.
        
        '''
        if(which == 'label'):
            self._params.fontsize_label = fontsize
        elif(which == 'tick'):
            self._params.fontsize_tick = fontsize
        elif(which == 'legend'):
            self._params.fontsize_legend = fontsize
        else:
            raise ValueError("set_fontsize: which must be one of the following: 'label', 'tick' or 'legend'")

    @staticmethod
    def _print_params_panels(key, vals, nrows, ncols):
        k = 0
        for i in range(nrows):
            line = "{:16s}: ".format(key) if i == 0 else "{:16s}  ".format("")
            for j in range(ncols):
                line += "{:s}, ".format(str(vals[k]))
                k = k + 1
            print(line)

    def add_legend(self, legend, which='upper right', outside=None, loc=None):
        '''
        Add customized legend object to specified axes.
        '''

        err_msg = "FrameTopLevel.add_legend() arg 'which' must be an int, a 2-tuple or one of the strings ['upper left', 'upper right', 'lower left', 'lower right']"
        codes = {'upper left': (0, 0),
                'upper right': (0, -1),
                'lower left': (-1, 0),
                'lower right': (-1, -1)
        }

        # which determines the ax to draw the legend
        if(isinstance(which, str)):
            which = codes.get(which)
            if(which is None):
                raise KeyError(err_msg)
        if(isinstance(which, tuple) or isinstance(which, list)):
            assert(len(which) == 2), err_msg
            which = self.__class__._parse_index(which, self.dims[0], self.dims[1])[0]
        elif(not isinstance(which, int)):
            raise ValueError(err_msg)

        if(loc is not None):
            legend.loc = loc
        if(outside is not None):
            legend.outside = outside

        self.legends.append({'legend':legend, 'which':which})
        
    def parameters(self, level="all"):
        '''
        Display parameters at a specified level.
        Parameters controls different levels of the figure. The top level is 
        'figure' (e.g. figsize, top, bottom), which defines the global layout 
        of the figure. The next level is 'panel', which defines number of 
        panels as well as the properties of EACH panel (e.g., dims, npanels, 
        xlabels(npanels), xticks(npanels)). The last level is 'ax' (TODO), 
        which fine controls plot elements in each ax of the figure.

        Parameters
        ----------
        level: string. Default = "all"
            If level == 'all', show all parameters
            If level == 'fig' or 'figure', show only figure level parameters
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
        
    def draw(self, verbose=False, sketch=False):
        '''
        Create canvas and draw the defined layout on the canvas.

        Lazy computation, the last step.

        Returns
        -------
        fig, axs

        '''
        
        # Figure Level
        pars = self._params
        fig = plt.figure(pars.num, figsize=pars.figsize, dpi=pars.dpi)

        # Frame Level
        axs = []
        gs = gridspec.GridSpec(
            pars.dims[0], pars.dims[1],
            height_ratios = pars.height_ratios,
            width_ratios = pars.width_ratios)
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
            else: # default ticks, we can still format them
                axs[i].set_xticklabels(
                    self.gen_ticklabels(axs[i].get_xticks(), self.xtickformat[i]),
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
            else: # default ticks, we can still format them
                axs[i].set_yticklabels(
                    self.gen_ticklabels(axs[i].get_yticks(), self.ytickformat[i]),
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
                            bottom=pars.bottom, top=pars.top,
                            left=pars.left, right=pars.right)
        fig.suptitle(pars.suptitle)
        # legends
        for lgd in self.legends:
            lgd['legend'].set_axes(axs[lgd['which']])
            lgd['legend'].fontsize = self._params.fontsize_legend
            lgd['legend'].draw()
        
        if(verbose == True):
            pars.show() # Show parameters
        return fig, axs

    def sketch(self):
        '''
        Quickly examine the current layout.
        '''
        
        self.draw(verbose=True, sketch=True)
        plt.show()

class FrameMulti(FrameTopLevel):
    '''
    Class specific for mutli-panel figures.


    A most useful derived class, FrameMulti, is specific for making multi-panel 
    plots. One example here is to create a 2x2 figure with tight layout:

           +-------+-------+
           |       |       |
         y |       |       |
           |       |       |
           +-------+-------+
           |       |       |
         y |       |       |
           |       |       |
           +-------+-------+
               x       x

        frm = FrameMulti(2,2,tight_layout=True)
        frm.set_xlabels('x', which='row')
        frm.set_ylabels('y', which='col')

    To see other examples, call frm.demo()

    A main method to customize the plots, e.g., setting axis ranges and labels, 
    use the class method in the following form:

        frm.set_[attributes]([attribute], which, erase)

    where, the 'which' argument specifies the panels to apply the function. 
    For example, which = 'bottom' will reset all the bottom panels; which = 
    [(1,2), (2,3)] will reset the attribute only for the two panels indicated.
    For details, see help(frm.set_panels_attributes).

    To set other parameters, one can use frm.parameters() to see a list of 
    accepted parameters, and use 
        
        frm.set_param(param, value) 

    to set certain parameters.

    While customizing the figure, one can always examine what the layout will 
    look like by using

        frm.sketch()

    This function will only show the layout of the final figure without 
    plotting any content to the axes.

    Finally, the figure will be drawn with:
    
        fig, axs = frm.draw()

    This will draw the frames on the canvas and returns a list of axs, which 
    can be passed to other plotting routines as an argument.

    '''
    
    def __init__(self, nrows, ncols, tight_layout=True):
        '''
        Parameters
        ----------
        nrows: int.
            Total number of rows, i.e., the number of panels in each column.
        
        ncols: int.
            Total number of columns, i.e., the number of panels in each row.

        tight_layout: boolean. Default = True.
            By default, create multi-panel figure with no space between two 
            panels and keep only the axis labels and ticklabels for the left 
            and bottom panels.
        '''
        
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

    @staticmethod
    def demo():
        '''
        This is a demo for some commonly used layouts for publications.

        I. 2 x 2, tight layout, identical panels

           +-------+-------+
           |       |       |
         y |       |       |
           |       |       |
           +-------+-------+
           |       |       |
         y |       |       |
           |       |       |
           +-------+-------+
               x       x

        >>> frm = FrameMulti(2,2,tight_layout=True)
        >>> frm.set_xlabels('x', which='row')
        >>> frm.set_ylabels('y', which='col')

        II. 2 x 2, independent panels

           +-------+    +-------+
           |       |    |       |
         y |       |  y |       |
           |       |    |       |
           +-------+    +-------+
               x            x
           +-------+    +-------+
           |       |    |       |
         y |       |  y |       |
           |       |    |       |
           +-------+    +-------+
               x            x

        >>> frm = FrameMulti(2,2,tight_layout=False)
        >>> frm.set_param('hspace', 0.25)
        >>> frm.set_xlabels('x')
        >>> frm.set_ylabels('y') # which = 'all' by default
        >>> frm.sketch()

        III. Main and side panels

            +-------+---+
            |       |   |
         y1 |       |   |
            |       |   |
            +-------+---+
         y2 |       | x
            +-------+
                x

        >>> frm = FrameMulti(2,2)
        >>> frm._params.height_ratios = [4, 1]
        >>> frm._params.width_ratios = [4, 1]
        >>> frm.set_xlabels('x', which=[(1,0),(0,1)])
        >>> frm.set_ylabels('y1', which=(0,0))
        >>> frm.set_ylabels('y2', which=(1,0))
        >>> frm.axisON[3] = False
        >>> frm.sketch()

        IV. (2) x 3 panels

            +-------+-------+-------+
            |       |       |       |
         y1 |       |       |       |
            |       |       |       |
            |       |       |       |
            +-------+-------+-------+
         y2 |       |       |       |
            +-------+-------+-------+
               x        x       x

        >>> frm = FrameMulti(2,3,tight_layout=True)
        >>> frm._params.height_ratios = [4, 1]
        >>> frm.set_xlabels('x', which='bottom')
        >>> frm.set_ylabels('y1', which=(0,0))
        >>> frm.set_ylabels('y2', which=(1,0))
        >>> frm.sketch()

        V. 2 x 2, tight layout with legends

           +-------+-------+ 111
           |       |       | 111
         y |       |       |
           |       |       |
           +-------+-------+
           |    333|       |
         y |       |       |
           |       |       | 2222
           +-------+-------+ 2222
               x       x

        >>> frm = FrameMulti(2,2, True)
        >>> frm.set_xlabels('xlabel')
        >>> frm.set_ylabels('ylabel')

        >>> lgd1 = Legend()
        >>> lgd1.add_line("lgd1:black line")
        >>> frm.add_legend(lgd1, which="upper right", loc="upper right")

        >>> lgd2 = Legend()
        >>> lgd2.add_patch("lgd2:red patch", fc='red')
        >>> frm.add_legend(lgd2, which="lower right", loc="lower right")

        >>> lgd3 = Legend()
        >>> lgd3.add_line("lgd3:thick blue dashed line", "blue", "--", 2)
        >>> frm.add_legend(lgd3, which="lower left", loc="upper right")

        >>> frm.set_param('right', 0.80)
        >>> frm.sketch()
        '''
        print(FrameMulti.demo.__doc__)

# plt.show changes figure dimension

# import configparser

# config_file = "multiframe.cfg"
# cfg = configparser.ConfigParser(inline_comment_prefixes=('#'))
# cfg.optionxform=str
# cfg.read(config_file)
